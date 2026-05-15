# Copyright 2026 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Server library utilities for Tiering Service."""

from collections.abc import Sequence
import datetime
from absl import logging
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import sqlalchemy.orm


def db_asset_to_proto(db_asset: db_schema.Asset) -> tiering_service_pb2.Asset:
  """Converts a db_schema.Asset to a tiering_service_pb2.Asset."""
  proto_asset = tiering_service_pb2.Asset()
  proto_asset.uuid = db_asset.asset_uuid
  proto_asset.path = db_asset.path
  proto_asset.user = db_asset.user
  if db_asset.tags:
    proto_asset.tags.extend(db_asset.tags)
  proto_asset.state = db_asset.state.value

  if db_asset.created_at:
    proto_asset.created_at.FromDatetime(db_asset.created_at)
  if db_asset.finalized_at:
    proto_asset.finalized_at.FromDatetime(db_asset.finalized_at)
  if db_asset.deleted_at:
    proto_asset.deleted_at.FromDatetime(db_asset.deleted_at)
  if db_asset.updated_at:
    proto_asset.updated_at.FromDatetime(db_asset.updated_at)

  for tp in db_asset.tier_paths:
    proto_tp = proto_asset.tier_paths.add()
    proto_tp.id = tp.id
    proto_tp.path = tp.path
    if tp.ready_at:
      proto_tp.ready_at.FromDatetime(tp.ready_at)
    if tp.expires_at:
      proto_tp.expires_at.FromDatetime(tp.expires_at)

    sb = tp.storage_backend
    proto_sb = proto_tp.storage_backend
    proto_sb.id = sb.id
    proto_sb.level = sb.level
    proto_sb.backend_type = sb.backend_type.value
    proto_sb.prefix = sb.prefix
    if sb.zone:
      proto_sb.zone = sb.zone
    elif sb.region:
      proto_sb.region = sb.region
    elif sb.multi_regions:
      proto_sb.multi_regions.regions.extend(sb.multi_regions)

  return proto_asset


async def find_backend_by_level(
    session: AsyncSession,
    level: int = 0,
) -> Sequence[db_schema.StorageBackend]:
  """Queries backends with the given level from the database."""
  result = await session.execute(
      select(db_schema.StorageBackend).filter_by(level=level)
  )
  return result.scalars().all()


def locate_closest_backend(
    backends: Sequence[db_schema.StorageBackend],
    zone: str | None,
    region: str | None,
) -> db_schema.StorageBackend | None:
  """Selects the closest storage backend matching input location metrics."""
  if zone:
    # Match the exact zone
    for backend in backends:
      if backend.zone == zone:
        return backend

    # When no region specified, match the zone prefix with the region
    if not region:
      for backend in backends:
        if backend.region and zone.startswith(backend.region):
          return backend
        if backend.multi_regions and any(
            zone.startswith(mr) for mr in backend.multi_regions
        ):
          return backend

  if region:
    for backend in backends:
      if backend.region == region:
        return backend
      if backend.multi_regions and region in backend.multi_regions:
        return backend

  return None


async def fetch_asset_by_identifier(
    session: AsyncSession,
    asset_uuid: str | None = None,
    path: str | None = None,
    inclusive_filter: Sequence[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets using optional UUID or path identifiers with state filtering."""
  if not asset_uuid and not path:
    logging.warning("No uuid or path specified")
    return []

  stmt = select(db_schema.Asset).options(
      sqlalchemy.orm.selectinload(db_schema.Asset.tier_paths).selectinload(
          db_schema.TierPath.storage_backend
      )
  )

  if asset_uuid:
    stmt = stmt.filter_by(asset_uuid=asset_uuid)
  elif path:
    stmt = stmt.filter_by(path=path).order_by(db_schema.Asset.created_at.desc())

  if inclusive_filter is not None:
    stmt = stmt.filter(db_schema.Asset.state.in_(inclusive_filter))

  result = await session.execute(stmt)
  return result.scalars().all()


async def fetch_asset_by_path(
    session: AsyncSession,
    path: str,
    inclusive_filter: Sequence[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets by path with optional state eligibility constraints."""
  return await fetch_asset_by_identifier(
      session, path=path, inclusive_filter=inclusive_filter
  )


async def fetch_asset_by_uuid(
    session: AsyncSession,
    asset_uuid: str,
    inclusive_filter: Sequence[db_schema.AssetState] | None = None,
) -> Sequence[db_schema.Asset]:
  """Fetches assets by UUID with optional state eligibility constraints."""
  return await fetch_asset_by_identifier(
      session, asset_uuid=asset_uuid, inclusive_filter=inclusive_filter
  )


def calculate_write_expires_at(
    interval_seconds: int,
) -> datetime.datetime:
  """Calculates a new write_expires_at timestamp with a 20% grace period buffer."""
  grace_buffer = int(interval_seconds * 0.2)
  total_seconds = interval_seconds + grace_buffer
  return datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  ) + datetime.timedelta(seconds=total_seconds)


async def create_or_fetch_asset(
    session: AsyncSession,
    request: tiering_service_pb2.ReserveRequest,
    backend: db_schema.StorageBackend,
    config: tiering_service_pb2.ServerConfig,
) -> db_schema.Asset:
  """Creates new asset or fetches existing on unique constraints conflict."""
  db_asset = db_schema.Asset(
      path=request.path,
      user=request.user,
      tags=list(request.tags) if request.tags else [],
      state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
      write_expires_at=calculate_write_expires_at(
          config.client_keep_alive_interval_seconds
      ),
  )
  storage_path = f"{backend.prefix.rstrip('/')}/{request.path.lstrip('/')}"
  tp = db_schema.TierPath(
      storage_backend=backend,
      path=storage_path,
  )
  db_asset.tier_paths.append(tp)

  try:
    session.add(db_asset)
    await session.commit()
    try:
      # Avoid ORM tracking that causes subsequent queries from the same session
      session.expunge(db_asset)
    except InvalidRequestError:
      pass

    # Fetch the asset again to load DB updated fields such as updated_at.
    res_list = await fetch_asset_by_uuid(
        session,
        db_asset.asset_uuid,
        inclusive_filter=[db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE],
    )
    res = res_list[0] if res_list else None
    assert res is not None
    return res
  except IntegrityError:
    await session.rollback()

  logging.info(
      "Reserve: Asset path already exists, fetching existing record: %s",
      request.path,
  )
  active_assets = await fetch_asset_by_path(
      session,
      request.path,
      inclusive_filter=[
          db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
          db_schema.AssetState.ASSET_STATE_STORED,
      ],
  )
  active_asset = active_assets[0] if active_assets else None
  if not active_asset:
    # This shouldn't happen rarely unless the asset was deleted after the
    # insert attempt.
    raise ValueError("Failed to retrieve reserved asset.")
  return active_asset


async def reserve_keep_alive(
    session: AsyncSession,
    uuid_val: str,
    interval_seconds: int,
) -> db_schema.Asset | None:
  """Extends the client writing keep alive expiration timestamp for an asset."""
  db_assets = await fetch_asset_by_uuid(session, uuid_val)
  db_asset = db_assets[0] if db_assets else None
  if not db_asset:
    return None

  db_asset.write_expires_at = calculate_write_expires_at(interval_seconds)
  await session.commit()
  return db_asset


async def finalize_asset(
    session: AsyncSession,
    asset_uuid: str,
) -> db_schema.Asset | None:
  """Finalizes registration status, transitions state to STORED inside a transaction."""
  stmt = (
      sqlalchemy.update(db_schema.Asset)
      .where(db_schema.Asset.asset_uuid == asset_uuid)
      .where(
          db_schema.Asset.state == db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE
      )
      .values(
          state=db_schema.AssetState.ASSET_STATE_STORED,
          finalized_at=datetime.datetime.now(datetime.timezone.utc).replace(
              tzinfo=None
          ),
          write_expires_at=None,
      )
  )

  result = await session.execute(stmt)
  await session.commit()

  # If 0 rows were updated, it indicates a conflict or already finalized!
  if result.rowcount == 0:
    raise ValueError("No matching asset found to finalize.")

  res = await fetch_asset_by_uuid(session, asset_uuid)
  if not res:
    # This should not happen, but checking just in case.
    raise RuntimeError("Asset not found after successful finalize operation.")
  return res[0]


async def delete_asset(
    session: AsyncSession,
    uuid: str | None = None,
    path: str | None = None,
) -> bool:
  """Sets asset status to DELETED in database context."""
  db_assets = await fetch_asset_by_identifier(session, uuid, path)
  db_asset = db_assets[0] if db_assets else None
  if not db_asset:
    return False

  db_asset.state = db_schema.AssetState.ASSET_STATE_DELETED
  db_asset.deleted_at = datetime.datetime.now(datetime.timezone.utc).replace(
      tzinfo=None
  )
  await session.commit()
  return True
