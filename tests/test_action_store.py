"""Action idempotency: the same action on the same target is stored once, whatever the wording."""

from fleet_copilot.db.action_store import ActionStore
from fleet_copilot.tools.action_tools import ACTION_REGISTRY


def _record(store: ActionStore, action: str, arguments: dict, tenant: str = "acme-001"):
    spec = ACTION_REGISTRY[action]
    return store.record(
        tenant_id=tenant,
        action_type=action,
        arguments=arguments,
        identity=spec.identity_of(arguments),
        justification="test",
        evidence_ids=["X@20260612T0000Z"],
        proposal_id="prop-1",
        approved_by="alice",
        run_id="run-1",
    )


def test_reworded_replacement_flag_is_a_duplicate(tmp_path):
    store = ActionStore(tmp_path / "actions.db")
    first = _record(
        store, "flag_device_for_replacement", {"device_id": "D1", "reason": "Battery worn"}
    )
    again = _record(
        store, "flag_device_for_replacement", {"device_id": "D1", "reason": "EOL battery"}
    )
    assert first.created and not again.created
    assert again.action_id == first.action_id


def test_different_component_or_tenant_is_a_new_action(tmp_path):
    store = ActionStore(tmp_path / "actions.db")
    ram = {"device_id": "D1", "component": "ram", "spec": "16 GB"}
    assert _record(store, "create_upgrade_order", ram).created
    assert not _record(store, "create_upgrade_order", {**ram, "spec": "32 GB"}).created
    assert _record(store, "create_upgrade_order", {**ram, "component": "storage"}).created
    assert _record(store, "create_upgrade_order", ram, tenant="globex-002").created
