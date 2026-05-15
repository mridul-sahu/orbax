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

import datetime
import unittest

from absl.testing import absltest
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server_lib
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker

from google.protobuf import timestamp_pb2


class ServerLibTest(absltest.TestCase):

  def test_db_asset_to_proto(self):
    dt_created = datetime.datetime(
        2026, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc
    )
    expected_ts_created = timestamp_pb2.Timestamp()
    expected_ts_created.FromDatetime(dt_created)

    tz_est = datetime.timezone(datetime.timedelta(hours=-5))
    dt_finalized = datetime.datetime(2026, 1, 2, 11, 0, 0, tzinfo=tz_est)
    expected_ts_finalized = timestamp_pb2.Timestamp()
    expected_ts_finalized.FromDatetime(dt_finalized)

    tz_pst = datetime.timezone(datetime.timedelta(hours=-8))
    dt_deleted = datetime.datetime(2026, 1, 3, 12, 0, 0, tzinfo=tz_pst)
    expected_ts_deleted = timestamp_pb2.Timestamp()
    expected_ts_deleted.FromDatetime(dt_deleted)

    tz_jst = datetime.timezone(datetime.timedelta(hours=9))
    dt_updated = datetime.datetime(2026, 1, 4, 13, 0, 0, tzinfo=tz_jst)
    expected_ts_updated = timestamp_pb2.Timestamp()
    expected_ts_updated.FromDatetime(dt_updated)

    tz_ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    dt_ready = datetime.datetime(2026, 1, 5, 14, 0, 0, tzinfo=tz_ist)
    expected_ts_ready = timestamp_pb2.Timestamp()
    expected_ts_ready.FromDatetime(dt_ready)

    tz_cest = datetime.timezone(datetime.timedelta(hours=2))
    dt_expires = datetime.datetime(2026, 1, 6, 15, 0, 0, tzinfo=tz_cest)
    expected_ts_expires = timestamp_pb2.Timestamp()
    expected_ts_expires.FromDatetime(dt_expires)

    db_backend = db_schema.StorageBackend(
        id=1,
        level=0,
        backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
        prefix="/mnt/lustre",
        zone="us-central1-a",
    )
    db_tp = db_schema.TierPath(
        id=10,
        path="/mnt/lustre/test/path",
        ready_at=dt_ready,
        expires_at=dt_expires,
        storage_backend=db_backend,
    )
    db_asset = db_schema.Asset(
        asset_uuid="test-uuid",
        path="test/path",
        user="test-user",
        tags=["tag1", "tag2"],
        state=db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
        created_at=dt_created,
        finalized_at=dt_finalized,
        deleted_at=dt_deleted,
        updated_at=dt_updated,
        tier_paths=[db_tp],
    )
    proto_asset = server_lib.db_asset_to_proto(db_asset)

    self.assertEqual(proto_asset.uuid, "test-uuid")
    self.assertEqual(proto_asset.path, "test/path")
    self.assertEqual(proto_asset.user, "test-user")
    self.assertEqual(list(proto_asset.tags), ["tag1", "tag2"])
    self.assertEqual(
        proto_asset.state, tiering_service_pb2.ASSET_STATE_ACTIVE_WRITE
    )
    self.assertEqual(proto_asset.created_at, expected_ts_created)
    self.assertEqual(proto_asset.finalized_at, expected_ts_finalized)
    self.assertEqual(proto_asset.deleted_at, expected_ts_deleted)
    self.assertEqual(proto_asset.updated_at, expected_ts_updated)

    self.assertLen(proto_asset.tier_paths, 1)
    tp_proto = proto_asset.tier_paths[0]
    self.assertEqual(tp_proto.id, 10)
    self.assertEqual(tp_proto.path, "/mnt/lustre/test/path")
    self.assertEqual(tp_proto.ready_at, expected_ts_ready)
    self.assertEqual(tp_proto.expires_at, expected_ts_expires)

    sb_proto = tp_proto.storage_backend
    self.assertEqual(sb_proto.id, 1)
    self.assertEqual(sb_proto.level, 0)
    self.assertEqual(
        sb_proto.backend_type, tiering_service_pb2.BACKEND_TYPE_LUSTRE
    )
    self.assertEqual(sb_proto.prefix, "/mnt/lustre")
    self.assertEqual(sb_proto.zone, "us-central1-a")


class ServerLibDbTest(absltest.TestCase, unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    tmp_file = self.create_tempfile()
    self.engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_file.full_path}"
    )
    async with self.engine.begin() as conn:
      await conn.run_sync(db_schema.Base.metadata.create_all)
    self.session_maker = sessionmaker(
        self.engine, expire_on_commit=False, class_=AsyncSession
    )

  async def asyncTearDown(self) -> None:
    await self.engine.dispose()
    await super().asyncTearDown()

  async def test_find_backend_by_level_and_locate_closest(self):
    async with self.session_maker() as session:
      b1 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre1",
          zone="us-central1-a",
      )
      b2 = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre2",
          zone="us-central1-b",
      )
      b3 = db_schema.StorageBackend(
          level=1,
          backend_type=db_schema.BackendType.BACKEND_TYPE_GCS,
          prefix="gs://bucket",
          region="us-central1",
      )
      session.add_all([b1, b2, b3])
      await session.commit()

      backends = await server_lib.find_backend_by_level(session, level=0)
      self.assertLen(backends, 2)

      backends_l1 = await server_lib.find_backend_by_level(session, level=1)
      self.assertLen(backends_l1, 1)
      self.assertEqual(backends_l1[0].prefix, "gs://bucket")

      # Test locate closest backend
      match = server_lib.locate_closest_backend(backends, "us-central1-a", None)
      self.assertIsNotNone(match)
      self.assertEqual(match.prefix, "/mnt/lustre1")

      match_fallback = server_lib.locate_closest_backend(
          backends, "us-central1-c", "us-central1"
      )
      self.assertIsNone(match_fallback)

  async def test_create_or_fetch_asset_and_queries(self):
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      session.add(backend)
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="test/path",
          user="test-user",
          zone="us-central1-a",
          tags=["tag1"],
      )
      config = tiering_service_pb2.ServerConfig(
          client_keep_alive_interval_seconds=600
      )

      # Create asset
      asset = await server_lib.create_or_fetch_asset(
          session, request, backend, config
      )
      self.assertEqual(asset.path, "test/path")
      self.assertLen(asset.tier_paths, 1)

      # Try creating it again (triggers unique conflict fetch fallback)
      asset2 = await server_lib.create_or_fetch_asset(
          session, request, backend, config
      )
      self.assertEqual(asset2.asset_uuid, asset.asset_uuid)

      # Query active asset by path
      fetched_assets = await server_lib.fetch_asset_by_path(
          session,
          "test/path",
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
              db_schema.AssetState.ASSET_STATE_STORED,
          ],
      )
      self.assertLen(fetched_assets, 1)
      self.assertEqual(fetched_assets[0].asset_uuid, asset.asset_uuid)

      # Query asset by uuid
      fetched_uuids = await server_lib.fetch_asset_by_uuid(
          session, asset.asset_uuid
      )
      self.assertLen(fetched_uuids, 1)
      self.assertEqual(fetched_uuids[0].asset_uuid, asset.asset_uuid)

      # Query asset by identifier (uuid or path)
      fetched_ids = await server_lib.fetch_asset_by_identifier(
          session, asset.asset_uuid, ""
      )
      self.assertLen(fetched_ids, 1)
      self.assertEqual(fetched_ids[0].asset_uuid, asset.asset_uuid)

  async def test_mutations_keep_alive_finalize_delete(self):
    async with self.session_maker() as session:
      backend = db_schema.StorageBackend(
          level=0,
          backend_type=db_schema.BackendType.BACKEND_TYPE_LUSTRE,
          prefix="/mnt/lustre",
          zone="us-central1-a",
      )
      session.add(backend)
      await session.commit()

      request = tiering_service_pb2.ReserveRequest(
          path="mut/path",
          user="test-user",
          zone="us-central1-a",
      )
      config = tiering_service_pb2.ServerConfig(
          client_keep_alive_interval_seconds=600
      )
      asset = await server_lib.create_or_fetch_asset(
          session, request, backend, config
      )

      # Keep alive
      updated = await server_lib.reserve_keep_alive(
          session, asset.asset_uuid, 1200
      )
      self.assertIsNotNone(updated)

      # Finalize
      finalized = await server_lib.finalize_asset(session, asset.asset_uuid)
      self.assertIsNotNone(finalized)
      assert finalized is not None
      self.assertEqual(finalized.state, db_schema.AssetState.ASSET_STATE_STORED)

      # Delete
      success = await server_lib.delete_asset(session, asset.asset_uuid, "")
      self.assertTrue(success)

      # Verify status is now DELETED
      session.expunge_all()
      # active asset fetch should be None now
      deleted_active = await server_lib.fetch_asset_by_path(
          session,
          "mut/path",
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
              db_schema.AssetState.ASSET_STATE_STORED,
          ],
      )
      self.assertEmpty(deleted_active)


if __name__ == "__main__":
  absltest.main()
