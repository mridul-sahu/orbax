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

"""Checkpoint Tiering Service (CTS) Server implementation."""

import asyncio
from collections.abc import Sequence
from concurrent import futures
import os
import sys

from absl import logging
import fire
import grpc
from orbax.checkpoint.experimental.tiering_service import db_lib
from orbax.checkpoint.experimental.tiering_service import db_schema
from orbax.checkpoint.experimental.tiering_service import server_config
from orbax.checkpoint.experimental.tiering_service import server_lib
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2
from orbax.checkpoint.experimental.tiering_service.proto import tiering_service_pb2_grpc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
import uvloop

_BEARER_PREFIX = "Bearer "


async def _get_oauth_token(context: grpc.aio.ServicerContext) -> str | None:
  """Extracts OAuth token from gRPC metadata."""
  logging.debug("Extracting OAuth token from metadata")
  metadata = dict(context.invocation_metadata())
  # Standard header for OAuth tokens in gRPC is 'authorization'
  auth_header = metadata.get("authorization")

  if auth_header is None:
    logging.warning("No authorization header found")
    return None

  if not auth_header.startswith(_BEARER_PREFIX):
    logging.warning("Malfomed authorization header: %s", auth_header)
    return None

  logging.debug("Found authorization header: %s", auth_header)
  return auth_header[len(_BEARER_PREFIX) :]


async def _verify_gcs_permissions(
    token: str | None, path: str, permissions: Sequence[str]
) -> bool:
  """Verifies if the caller has necessary permissions on GCS."""
  logging.info("Verifying GCS permissions for path: %s", path)
  logging.debug("Requested permissions: %s", permissions)
  # TODO: b/503445654 - Implement actual IAM permission verification
  # For now, return True if a token is provided, and False otherwise,
  # to allow testing PERMISSION_DENIED errors.
  logging.debug("Permission check result: %s", token is not None)
  return token is not None


async def _has_write_permission(token: str | None, path: str) -> bool:
  """Checks whether bearer token possesses permission scopes for storage write."""
  return await _verify_gcs_permissions(token, path, ["storage.objects.create"])


def _has_location(request) -> bool:
  """Checks whether request specifies a location (zone or region)."""
  return request.HasField("zone") or request.HasField("region")


def _has_identifier(request) -> bool:
  """Checks whether request has either uuid or path identifier."""
  return request.HasField("uuid") or request.HasField("path")


class TieringServiceServicer(tiering_service_pb2_grpc.TieringServiceServicer):
  """Servicer for the TieringService."""

  def __init__(self, config: tiering_service_pb2.ServerConfig):
    super().__init__()
    self._config = config
    self._engine = db_lib.get_async_engine(self._config)
    self._session_maker = sessionmaker(
        self._engine,
        # Required for async session usage
        expire_on_commit=False,
        class_=AsyncSession,
    )
    self._level0_backends = None

  async def initialize(self) -> None:
    """Initializes the servicer, loading static data like level 0 backends."""
    async with self._session_maker() as session:
      self._level0_backends = await server_lib.find_backend_by_level(
          session, level=0
      )

  async def _run_async_db(self, coro_fn):
    async with self._session_maker() as session:
      res = await coro_fn(session)
      session.expunge_all()
      return res

  async def Reserve(
      self,
      request: tiering_service_pb2.ReserveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveResponse:
    """Reserves a new asset or looks up an existing one."""
    logging.info("Reserve requested for path: %s", request.path)
    token = await _get_oauth_token(context)
    if not await _has_write_permission(token, request.path):
      logging.warning("Permission denied for Reserve on path: %s", request.path)
      await context.abort(
          grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
      )
      return tiering_service_pb2.ReserveResponse()

    if not _has_location(request):
      logging.error(
          "Reserve: No location specified for path: %s, user: %s",
          request.path,
          request.user,
      )
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No zone or region specified"
      )
      return tiering_service_pb2.ReserveResponse()

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.ReserveResponse:
      assert self._level0_backends is not None, "Servicer not initialized"
      backends = self._level0_backends
      backend = server_lib.locate_closest_backend(
          backends, request.zone, request.region
      )
      if not backend:
        zone_val = request.zone if request.HasField("zone") else ""
        region_val = request.region if request.HasField("region") else ""
        await context.abort(
            grpc.StatusCode.INTERNAL,
            f"No level 0 storage backend found for zone:{zone_val} /"
            f" region:{region_val}",
        )
        return tiering_service_pb2.ReserveResponse()

      try:
        db_asset = await server_lib.create_or_fetch_asset(
            session, request, backend, self._config
        )
      except ValueError as e:
        await context.abort(grpc.StatusCode.INTERNAL, str(e))
        return tiering_service_pb2.ReserveResponse()

      return tiering_service_pb2.ReserveResponse(
          asset=server_lib.db_asset_to_proto(db_asset),
          keep_alive_interval_seconds=self._config.client_keep_alive_interval_seconds,
      )

    return await self._run_async_db(_coro)

  async def ReserveKeepAlive(
      self,
      request: tiering_service_pb2.ReserveKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.ReserveKeepAliveResponse:
    """Extends the writing timeout for an asset."""
    logging.info("ReserveKeepAlive requested for UUID: %s", request.uuid)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.ReserveKeepAliveResponse:
      db_asset = await server_lib.reserve_keep_alive(
          session, request.uuid, self._config.client_keep_alive_interval_seconds
      )
      if not db_asset:
        logging.warning("ReserveKeepAlive: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.ReserveKeepAliveResponse()

      return tiering_service_pb2.ReserveKeepAliveResponse(
          keep_alive_interval_seconds=self._config.client_keep_alive_interval_seconds,
      )

    return await self._run_async_db(_coro)

  async def Finalize(
      self,
      request: tiering_service_pb2.FinalizeRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.FinalizeResponse:
    """Finalizes an asset, moving it to STORED state."""
    logging.info("Finalize requested for UUID: %s", request.uuid)
    token = await _get_oauth_token(context)

    async def _coro(
        session: AsyncSession,
    ) -> tiering_service_pb2.FinalizeResponse:
      db_assets = await server_lib.fetch_asset_by_uuid(
          session,
          request.uuid,
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
          ],
      )
      db_asset = db_assets[0] if db_assets else None
      if not db_asset:
        logging.warning("Finalize: Asset not found: %s", request.uuid)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.FinalizeResponse()

      assert db_asset is not None
      # Verify write permission before finalizing
      if not await _has_write_permission(token, db_asset.path):
        logging.warning(
            "Permission denied for Finalize on path: %s", db_asset.path
        )
        await context.abort(
            grpc.StatusCode.PERMISSION_DENIED, "Insufficient GCS permissions"
        )
        return tiering_service_pb2.FinalizeResponse()

      try:
        db_asset = await server_lib.finalize_asset(session, request.uuid)
        assert db_asset is not None
      except ValueError as e:
        logging.warning("Finalize: %s", str(e))
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        return tiering_service_pb2.FinalizeResponse()

      return tiering_service_pb2.FinalizeResponse(
          asset=server_lib.db_asset_to_proto(db_asset)
      )

    return await self._run_async_db(_coro)

  async def Prefetch(
      self,
      request: tiering_service_pb2.PrefetchRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchResponse:
    """Signals CTS to copy an asset to Tier 0 storage."""
    if not _has_location(request):
      await context.abort(
          grpc.StatusCode.INVALID_ARGUMENT, "No location specified"
      )
      return tiering_service_pb2.PrefetchResponse()
    # TODO: b/503445654 - Trigger async copy to closest storage tier to user.

    await context.abort(
        grpc.StatusCode.UNIMPLEMENTED, "Prefetch Not Implemented"
    )
    return tiering_service_pb2.PrefetchResponse()

  async def PrefetchKeepAlive(
      self,
      request: tiering_service_pb2.PrefetchKeepAliveRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.PrefetchKeepAliveResponse:
    """Signals that the client is still reading/waiting for promotion."""
    logging.info("PrefetchKeepAlive requested for UUID: %s", request.uuid)

    await context.abort(
        grpc.StatusCode.UNIMPLEMENTED, "PrefetchKeepAlive Not Implemented"
    )
    return tiering_service_pb2.PrefetchKeepAliveResponse()

  async def Delete(
      self,
      request: tiering_service_pb2.DeleteRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.DeleteResponse:
    """Deletes an asset from CTS tracking."""
    logging.info("Delete requested")

    await context.abort(grpc.StatusCode.UNIMPLEMENTED, "Delete Not Implemented")
    return tiering_service_pb2.DeleteResponse()

  async def Info(
      self,
      request: tiering_service_pb2.InfoRequest,
      context: grpc.aio.ServicerContext,
  ) -> tiering_service_pb2.InfoResponse:
    """Returns metadata about an asset."""
    identifier = request.uuid if request.HasField("uuid") else request.path
    logging.info("Info requested for identifier: %s", identifier)

    async def _coro(session: AsyncSession) -> tiering_service_pb2.InfoResponse:
      db_assets = await server_lib.fetch_asset_by_identifier(
          session,
          asset_uuid=request.uuid if request.HasField("uuid") else None,
          path=request.path if request.HasField("path") else None,
          inclusive_filter=[
              db_schema.AssetState.ASSET_STATE_ACTIVE_WRITE,
              db_schema.AssetState.ASSET_STATE_STORED,
          ],
      )
      if not db_assets:
        logging.warning("Info: Asset not found: %s", identifier)
        await context.abort(grpc.StatusCode.NOT_FOUND, "Asset not found")
        return tiering_service_pb2.InfoResponse()

      logging.debug("Returning info for assets: %s", db_assets)
      return tiering_service_pb2.InfoResponse(
          assets=[server_lib.db_asset_to_proto(asset) for asset in db_assets]
      )

    return await self._run_async_db(_coro)


async def setup_storage_backends(
    config: tiering_service_pb2.ServerConfig,
) -> None:
  """Initializes the database if uninitialized, otherwise verifies it matches configuration."""
  if not await db_lib.async_is_db_initialized(config):
    await db_lib.async_initialize_db(config)
  else:
    await db_lib.async_verify_db(config)


class CtsServer:
  """Checkpoint Tiering Service (CTS) Server CLI."""

  async def serve(self, yaml_path: str) -> None:
    """Starts the gRPC server.

    Args:
      yaml_path: Path to the YAML configuration file.
    """
    config = server_config.load_config(yaml_path)
    await setup_storage_backends(config)

    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = TieringServiceServicer(config)
    await servicer.initialize()
    tiering_service_pb2_grpc.add_TieringServiceServicer_to_server(
        servicer, server
    )

    server_creds = os.environ.get("SERVER_CREDS")  # pylint: disable=unused-variable

    server.add_secure_port("[::]:50051", server_creds)
    await server.start()

    # TODO(b/503445463): Start background garbage collection task to handle
    # expired assets.

    await server.wait_for_termination()


def main(argv: Sequence[str] | None = None) -> None:
  """Main entry point for CTS server."""
  if argv is None:
    argv = sys.argv
  uvloop.install()
  try:
    asyncio.get_event_loop()
  except RuntimeError:
    # Create the high-performance uvloop instead
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
  fire.Fire(CtsServer, command=argv[1:])


if __name__ == "__main__":
  main()
