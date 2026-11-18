"""Router tests: model classification, parsing, and the fallback chain.

All offline: the model is a scripted fake; no endpoint is contacted.
"""

from langchain_core.messages import AIMessage

from jarvis.agent.router import Router, heuristic_route, parse_route_decision
from jarvis.config import Settings
from jarvis.schemas import Route

from fakes import ScriptedChatModel


def make_router(script):
    return Router(Settings(), model=ScriptedChatModel(script=script))


def test_model_classification_picks_gmail():
    router = make_router([AIMessage(content='{"route": "gmail", "reason": "user asks about their inbox"}')])
    decision = router.classify("any unread mail from Priya?")
    assert decision.route == Route.GMAIL
    assert decision.source == "model"
    assert "inbox" in decision.reason


def test_model_classification_picks_research():
    router = make_router([AIMessage(content='{"route": "research", "reason": "needs a deep cited answer"}')])
    decision = router.classify("explain how inflation works")
    assert decision.route == Route.RESEARCH


def test_model_classification_picks_direct_answer():
    router = make_router([AIMessage(content='{"route": "direct_answer", "reason": "pure conversation"}')])
    decision = router.classify("what do you think about remote work?")
    assert decision.route == Route.DIRECT_ANSWER


def test_json_embedded_in_extra_text_still_parses():
    router = make_router([AIMessage(content='Sure! {"route": "spotify", "reason": "music request"} hope that helps')])
    decision = router.classify("play some lofi")
    assert decision.route == Route.SPOTIFY
    assert decision.source == "model"


def test_unparseable_model_answer_falls_back_to_keywords():
    router = make_router([AIMessage(content="I think you want the mail thing")])
    decision = router.classify("check my gmail inbox")
    assert decision.route == Route.GMAIL
    assert decision.source == "heuristic"


def test_unknown_route_value_falls_back():
    router = make_router([AIMessage(content='{"route": "teleport", "reason": "nope"}')])
    decision = router.classify("what is the weather today")
    assert decision.route == Route.WEB_SEARCH
    assert decision.source == "heuristic"


def test_model_error_falls_back_to_keywords():
    router = make_router([RuntimeError("model server unreachable")])
    decision = router.classify("schedule a meeting tomorrow")
    assert decision.route == Route.CALENDAR
    assert decision.source == "heuristic"


def test_no_match_anywhere_gives_direct_answer():
    router = make_router([RuntimeError("down")])
    decision = router.classify("hello there")
    assert decision.route == Route.DIRECT_ANSWER
    assert decision.source == "heuristic"


def test_schedule_a_post_is_social_not_calendar():
    decision = heuristic_route("schedule a post about our launch on instagram")
    assert decision.route == Route.SOCIAL_CONTENT


def test_parse_route_decision_rejects_garbage():
    assert parse_route_decision("") is None
    assert parse_route_decision("not json at all") is None
    assert parse_route_decision('{"route": "gmail"') is None
    assert parse_route_decision('{"reason": "no route key"}') is None


def test_router_never_raises_on_empty_script_model():
    router = make_router([AIMessage(content="")])
    decision = router.classify("sing me something on spotify")
    assert decision.route == Route.SPOTIFY
