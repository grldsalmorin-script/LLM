from agentic_ai_workflow.main import build_demo


def test_demo_flow_returns_frontend_event():
    jasmine, bus, async_handler = build_demo()
    command_id = jasmine.handle_user_message("company_abc", "reconcile transaction")

    bus.drain()

    assert async_handler.status_by_command_id[command_id] == "SUCCESS"
    assert jasmine.events[0]["event_name"] == "NicoleReconciliationReviewed"
    assert jasmine.events[0]["payload"]["candidate"]["requires_human_approval"] is True
