"""
Regression tests for AgentManager routing (Improvement Pass, Priority 1.1).

Mark X.6 fixed trigger-happy specialist routing by introducing weighted
scoring plus ROUTE_THRESHOLD; these tests pin that behaviour so it cannot
silently regress: primary signals dominate, weak lone keywords fall through
to the general ORION capacity, and ties resolve in registration order.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orion_core.agents import AgentManager, BaseAgent


class _Signal:
    def __init__(self):
        self.emitted = []

    def emit(self, *payload):
        self.emitted.append(payload)


class _StubBus:
    def __getattr__(self, name):
        sig = _Signal()
        object.__setattr__(self, name, sig)
        return sig


class _StubRouter:
    """No live provider: agents fall back to returning the specialist brief."""

    def has_text_fallback(self):
        return False


def _manager():
    return AgentManager(_StubRouter(), _StubBus())


class _NicheAgent(BaseAgent):
    name = "niche"
    title = "Niche Test Agent"
    expertise = "test fixture"
    primary = (r"\bzorbulator\b",)
    keywords = (r"\bzorb\b", r"\bflibber\b")


class _RivalAgent(BaseAgent):
    name = "rival"
    title = "Rival Test Agent"
    expertise = "test fixture"
    primary = (r"\bzorbulator\b",)
    keywords = ()


# ── weighted scoring mechanics ────────────────────────────────────────────────

def test_route_threshold_value_pinned():
    # Mark X.6 contract: a lone supporting keyword (weight 1) must never route.
    assert AgentManager.ROUTE_THRESHOLD == 3
    assert BaseAgent._PRIMARY_WEIGHT == 3
    assert BaseAgent._SECONDARY_WEIGHT == 1


def test_primary_signal_meets_threshold_and_routes():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    agent, score = mgr.route_with_confidence("engage the zorbulator")
    assert agent is not None and agent.name == "niche" and score == 3
    assert mgr.route("engage the zorbulator") is mgr._agents["niche"]


def test_lone_keyword_scores_but_does_not_route():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    agent, score = mgr.route_with_confidence("pass me the zorb")
    assert agent is not None and score == 1          # signal exists …
    assert mgr.route("pass me the zorb") is None     # … but below threshold


def test_two_keywords_still_below_threshold():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    _agent, score = mgr.route_with_confidence("the zorb and the flibber")
    assert score == 2
    assert mgr.route("the zorb and the flibber") is None


def test_primary_plus_keywords_accumulate():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))
    _agent, score = mgr.route_with_confidence("zorbulator with zorb and flibber")
    assert score == 5


def test_no_signal_returns_none_and_zero():
    mgr = _manager()
    agent, score = mgr.route_with_confidence("qqq zzz xxyzzy")
    assert agent is None and score == 0
    assert mgr.route("qqq zzz xxyzzy") is None


def test_ties_resolve_in_registration_order():
    mgr = _manager()
    mgr.register(_NicheAgent(mgr.router, mgr.bus))   # registered first
    mgr.register(_RivalAgent(mgr.router, mgr.bus))   # same primary pattern
    agent, score = mgr.route_with_confidence("engage the zorbulator")
    assert score == 3 and agent.name == "niche"


# ── real specialists: representative routing smoke ────────────────────────────

def test_coding_request_routes_to_coding_agent():
    agent = _manager().route("debug this python traceback for me")
    assert agent is not None and agent.name == "coding"


def test_marketing_request_routes_to_marketing_agent():
    agent = _manager().route("plan a marketing campaign for the newsletter")
    assert agent is not None and agent.name == "marketing"


def test_weak_lone_email_mention_falls_through():
    # The documented Mark X.6 case: "email" alone is a weight-1 keyword and
    # must not commandeer the turn away from the general persona.
    assert _manager().route("check my email please") is None


# ── dispatch entry point ──────────────────────────────────────────────────────

def test_dispatch_empty_request_refused():
    result = asyncio.run(_manager().dispatch(""))
    assert not result.ok


def test_dispatch_unknown_agent_lists_registry():
    result = asyncio.run(_manager().dispatch("do something", agent_name="nonexistent"))
    assert not result.ok
    assert "coding" in result.text and "marketing" in result.text


def test_dispatch_auto_with_no_signal_says_nothing_was_dispatched():
    # Mark XXXX.4: read as success, this no-op was described to the owner as a
    # security audit "currently running".
    result = asyncio.run(_manager().dispatch("qqq zzz xxyzzy", agent_name="auto"))
    assert not result.ok
    assert "nothing was dispatched" in result.text.lower()
    assert "do not say a task is running" in result.text.lower()


def test_dispatch_named_agent_bypasses_threshold():
    # Explicitly naming a specialist must work even with zero keyword signal.
    result = asyncio.run(_manager().dispatch("qqq zzz xxyzzy", agent_name="coding"))
    assert result.ok
    assert "Specialist brief" in result.text        # provider-less fallback path


def test_dispatch_blocks_destructive_request():
    import pytest
    from orion_core.security import SecurityViolation
    with pytest.raises(SecurityViolation):
        asyncio.run(_manager().dispatch("run rm -rf / now", agent_name="coding"))


def test_describe_lists_all_registered_specialists():
    rows = _manager().describe()
    names = {row["name"] for row in rows}
    assert {"marketing", "coding", "research"} <= names
    # Retired in Mark XXXI: chat personas with no instruments behind them.
    assert not names & {"design", "fashion", "entertainment"}


# ── Mark XXII: three new specialists fitting ORION's purpose ─────────────────

def test_new_specialists_are_registered():
    mgr = _manager()
    for name in ("neuroscience", "aiml", "security"):
        assert mgr.get(name) is not None, name


def test_neuroscience_question_routes_to_the_neuroscientist():
    mgr = _manager()
    pick = mgr.route("explain synaptic plasticity and how dopamine shapes the hippocampus")
    assert pick is not None and pick.name == "neuroscience"


def test_ml_question_routes_to_the_ml_agent():
    mgr = _manager()
    pick = mgr.route("how do I fine-tune a transformer in pytorch without overfitting")
    assert pick is not None and pick.name == "aiml"


def test_security_question_routes_to_the_security_agent():
    mgr = _manager()
    pick = mgr.route("how do I recognise a keylogger or rootkit and harden against ransomware")
    assert pick is not None and pick.name == "security"


def test_security_agent_is_defensive_and_educational():
    mgr = _manager()
    persona = mgr.get("security").persona.lower()
    assert "recognition" in persona or "defend" in persona
    assert "educational" in persona or "lawful" in persona
    assert "never weaponise" in persona or "no working attack" in persona


def test_neuroscience_persona_matches_the_users_field():
    mgr = _manager()
    persona = mgr.get("neuroscience").persona.lower()
    assert "neural engineering" in persona and "psychology" in persona


@pytest.mark.parametrize("question,name", [
    ("Diagnose why ORION stopped talking at 23:41", "reliability"),
    ("Prepare a runtime reliability incident report from the logs", "reliability"),
    ("Review the API integration contract and OAuth retry policy", "integrations"),
    ("Find the MCP JSON-RPC pagination problem", "integrations"),
    ("Pick up where we left off on the project", "continuity"),
    ("Resume my project and identify unfinished next steps", "continuity"),
])
def test_operational_specialists_route_clear_requests(question, name):
    agent = _manager().route(question)
    assert agent is not None and agent.name == name


@pytest.mark.parametrize("question", [
    "check my email please", "what is my next appointment", "token", "api", "logs",
    "deadline", "open the browser", "hello",
])
def test_operational_specialists_do_not_capture_weak_or_unrelated_mentions(question):
    agent = _manager().route(question)
    assert agent is None or agent.name not in {"reliability", "integrations", "continuity"}


@pytest.mark.parametrize("name,instrument", [
    ("reliability", "incident_lookup"),
    ("integrations", "capability_status"),
    ("continuity", "situation"),
])
def test_operational_specialists_consult_the_existing_factory_and_keep_evidence(name, instrument):
    from orion_core.data import ToolResult
    from orion_core.reasoning import BUDGETS, ReasoningEngine, ReasoningTier

    class _Profile:
        name = "scripted"

    class _Router:
        def __init__(self):
            self.calls = []

        def has_text_fallback(self):
            return True

        async def generate_text(self, prompt, **kwargs):
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return _Profile(), f'TOOL: {instrument}\nARGS: {{"query": "orion"}}'
            assert "Observed saved state" in prompt
            return _Profile(), "The observation identifies the next step."

    async def _observe(_name, _args):
        return ToolResult("Observed saved state")

    router, bus = _Router(), _StubBus()
    mgr = AgentManager(router, bus)
    engine = ReasoningEngine(bus, router, mgr, dispatch=_observe)
    mgr.attach_toolbelt_factory(lambda: engine.toolbelt(BUDGETS[ReasoningTier.STANDARD]))
    result = asyncio.run(mgr.dispatch("Review the known state", agent_name=name))
    assert result.ok and "identifies the next step" in result.text
    assert result.evidence and result.evidence[0]["tool"] == instrument
    assert result.evidence[0]["ok"] is True
    assert len(router.calls) == 2


# ── routing by meaning (Mark XXXX.7, 28 Oct 2026) ────────────────────────────
#
# The keyword router sent the right specialist 31/88 times on SEEN (written
# before the change) and 6/55 on UNSEEN (written after it, never used to
# tune it). Meaning-based routing: 81/88 and 43/55, with at most one
# small-talk request sent to a specialist.

ROUTING_SEEN = [
    ('How do I grow my TikTok shop sales this month?', ('marketing',)),
    ('Write me a content calendar for my Instagram brand', ('marketing',)),
    ("What's a good conversion funnel for an online course?", ('marketing',)),
    ('Improve the SEO on my product pages', ('marketing',)),
    ('Plan a Meta ads campaign with a 200 pound budget', ('marketing',)),
    ('Why is my click-through rate so low on my ads?', ('marketing',)),
    ('Come up with hooks for a launch video for my clothing line', ('marketing',)),
    ('How should I price my product so more people buy it?', ('marketing',)),
    ("Debug this Python traceback: KeyError 'name'", ('coding',)),
    ('Refactor this function so it runs faster', ('coding',)),
    ('Why does my React component render twice?', ('coding',)),
    ('Write a unit test for the parser', ('coding',)),
    ('My C++ program segfaults when I free the buffer', ('coding',)),
    ('How do I structure a Flask API project?', ('coding', 'integrations')),
    ('Fix the merge conflict in my git branch', ('coding',)),
    ('This loop is really slow, how can I speed it up?', ('coding',)),
    ('Compare the iPhone 16 and the Pixel 9 for me', ('research',)),
    ('Research the best UK universities for a neuroscience masters', ('research', 'neuroscience')),
    ('Give me a structured analysis of the pros and cons of remote work', ('research',)),
    ('Evaluate these three laptops for a student budget', ('research',)),
    ('Summarise the evidence on intermittent fasting', ('research',)),
    ('What are the main arguments for and against nuclear power?', ('research',)),
    ('Look into which electric car has the best range for the price', ('research',)),
    ('Investigate why house prices rose so much in 2021', ('research',)),
    ('Explain long-term potentiation in the hippocampus', ('neuroscience',)),
    ('How do action potentials travel along myelinated axons?', ('neuroscience',)),
    ("What's the role of dopamine in reward prediction error?", ('neuroscience',)),
    ('Help me revise the stages of sleep for my exam', ('neuroscience',)),
    ('How does working memory work in the prefrontal cortex?', ('neuroscience',)),
    ('What is synaptic plasticity?', ('neuroscience',)),
    ('Explain classical versus operant conditioning', ('neuroscience',)),
    ('Why do we forget things, what does the brain actually do?', ('neuroscience',)),
    ('How do transformers use attention?', ('aiml',)),
    ('Should I fine-tune a 3B model with LoRA or do a full fine-tune?', ('aiml',)),
    ('Why is my model overfitting the training data?', ('aiml',)),
    ('Explain gradient descent simply', ('aiml',)),
    ('How do I evaluate a classifier on an imbalanced dataset?', ('aiml',)),
    ("What's the difference between RAG and fine-tuning?", ('aiml',)),
    ('How many GPUs do I need to train a small language model?', ('aiml',)),
    ('Explain what embeddings are', ('aiml',)),
    ('How can I tell if an email is a phishing attempt?', ('security',)),
    ('Is my Wi-Fi network secure enough?', ('security',)),
    ('Explain SQL injection and how to defend against it', ('security',)),
    ('Threat model my home server', ('security',)),
    ('What is ransomware and how do I protect my files from it?', ('security',)),
    ('Should I use a password manager?', ('security',)),
    ('Someone tried to log into my Instagram from another country, what do I do?', ('security',)),
    ('How does two-factor authentication stop account takeover?', ('security',)),
    ('Why does ORION freeze when I open the command deck?', ('reliability',)),
    ('ORION keeps crashing after startup, diagnose it', ('reliability',)),
    ('Why did your voice go silent earlier?', ('reliability',)),
    ("Your responses are lagging, what's causing it?", ('reliability',)),
    ('The app went not responding again, find the cause', ('reliability',)),
    ('Check your error logs and tell me what broke', ('reliability',)),
    ('Why did the globe page crash?', ('reliability',)),
    ('How reliable have you been this week?', ('reliability',)),
    ('Connect the Notion API to my tasks database', ('integrations',)),
    ('Why is the Gmail MCP server failing to authenticate?', ('integrations',)),
    ('How do I paginate through the Shopify REST API?', ('integrations',)),
    ('Set up a webhook from Stripe to my server', ('integrations',)),
    ('The OAuth token keeps expiring, how do I refresh it?', ('integrations', 'security')),
    ("I'm getting rate limit errors from an API, how should I retry?", ('integrations',)),
    ('Hook Home Assistant up to ORION', ('integrations',)),
    ('Should this integration use REST or GraphQL?', ('integrations', 'coding')),
    ('Where did we leave off on the ORION project?', ('continuity',)),
    ('Resume my dissertation work from last week', ('continuity',)),
    ("What were the unfinished tasks from yesterday's session?", ('continuity',)),
    ("Give me a handoff summary of what's left to do", ('continuity',)),
    ('Pick up where I stopped on the marketing plan', ('continuity', 'marketing')),
    ('What was I working on before the weekend?', ('continuity',)),
    ('Summarise progress so far and the next steps', ('continuity',)),
    ('Continue the project we started on Monday', ('continuity',)),
    ("What's filling up my C drive?", ('files',)),
    ('Tidy up my Downloads folder', ('files',)),
    ('Find all the PDFs from my lectures and put them in one folder', ('files',)),
    ("Clear out old installers I don't need", ('files',)),
    ('Which folders are the biggest on my laptop?', ('files',)),
    ('Organise my desktop files by type', ('files',)),
    ('Gather every photo from last summer into one folder', ('files',)),
    ('Delete the duplicate files in my Documents', ('files',)),
    ("What couldn't you do this week?", ('triage',)),
    ('What keeps failing for you?', ('triage', 'reliability')),
    ('What should we build next for ORION?', ('triage',)),
    ('Which of your tools fail most often?', ('triage', 'reliability')),
    ("List the capability gaps you've noticed", ('triage',)),
    ('What are you worst at right now?', ('triage',)),
    ('What would make you more useful to me?', ('triage',)),
    ('Which requests did you have to refuse recently?', ('triage',)),
    ('Good morning ORION', ()),
    ("What's the weather tomorrow?", ()),
    ('Play some music', ()),
    ('Tell me a joke', ()),
    ('How are you today?', ()),
    ('Set a timer for ten minutes', ()),
    ('What time is it in Tokyo?', ()),
    ('Open Spotify', ()),
    ("Thanks, that's all", ()),
    ("What's 15 percent of 240?", ()),
    ('Remind me to call mum at 6', ()),
    ('Turn the volume up', ()),
    ('Who made you?', ()),
    ('Goodnight', ()),
    ("What's on my calendar today?", ()),
]

ROUTING_UNSEEN = [
    ('my etsy store barely gets any views, any ideas to fix that', ('marketing',)),
    ('draft three captions for a gym clothing post', ('marketing',)),
    ('what budget split between google and facebook ads makes sense', ('marketing',)),
    ('is it worth paying influencers to promote my app', ('marketing',)),
    ('help me name my new skincare brand and write a tagline', ('marketing',)),
    ("NameError: name 'df' is not defined, what does that mean", ('coding',)),
    ('write a bash script that backs up a folder every night', ('coding',)),
    ('my java code throws a null pointer exception', ('coding',)),
    ('convert this javascript callback into async await', ('coding',)),
    ("why won't my rust program borrow this variable", ('coding',)),
    ('is a macbook air or a dell xps better for uni', ('research',)),
    ('what do studies say about caffeine and exam performance', ('research',)),
    ('find me the cheapest way to get from birmingham to london by train', ('research',)),
    ('analyse whether I should rent or buy a car as a student', ('research',)),
    ('what caused the 2008 financial crisis', ('research',)),
    ("what's the difference between the sympathetic and parasympathetic nervous system", ('neuroscience',)),
    ('quiz me on the parts of a neuron', ('neuroscience',)),
    ('how does the amygdala process fear', ('neuroscience',)),
    ('explain what an EEG measures', ('neuroscience',)),
    ('what does serotonin actually do', ('neuroscience',)),
    ("what's a convolutional neural network good for", ('aiml',)),
    ('how does reinforcement learning work', ('aiml',)),
    ("my loss isn't going down when I train, help", ('aiml',)),
    ('which open source LLM can I run on an 8GB GPU', ('aiml',)),
    ('explain the bias variance tradeoff', ('aiml',)),
    ('I clicked a dodgy link in a text, am I hacked', ('security',)),
    ('how do I check if my password was in a data breach', ('security',)),
    ('is it safe to use public wifi at the airport', ('security',)),
    ('explain cross-site scripting', ('security',)),
    ("what's the safest way to store my crypto keys", ('security',)),
    ('you went quiet in the middle of answering me, why', ('reliability',)),
    ('why does starting you up take so long', ('reliability',)),
    ('the face window flickered black again', ('reliability',)),
    ('you crashed twice today, find out why', ('reliability',)),
    ('why is your memory usage so high', ('reliability',)),
    ('can you pull my spotify listening history through their api', ('integrations',)),
    ('my notion integration says unauthorized', ('integrations',)),
    ('set up zapier style automation between gmail and notion', ('integrations',)),
    ('how do I get an access token for the twitter api', ('integrations',)),
    ('connect my philips hue bridge', ('integrations',)),
    ('what were we in the middle of before I went out', ('continuity',)),
    ('catch me up on the robot project', ('continuity',)),
    ("what's left to finish on my coursework", ('continuity',)),
    ("let's carry on from where we stopped", ('continuity',)),
    ('what did we get done yesterday', ('continuity',)),
    ('how much space do my videos take up', ('files',)),
    ('move all my screenshots into a screenshots folder', ('files',)),
    ('empty out the temp files clogging my pc', ('files',)),
    ("find the biggest files I haven't opened in a year", ('files',)),
    ('put my university notes into folders by module', ('files',)),
    ('what have you failed at most this month', ('triage', 'reliability')),
    ('which abilities should I give you next', ('triage',)),
    ("what do you wish you could do that you can't", ('triage',)),
    ("what have people asked you that you couldn't handle", ('triage',)),
    ("rank the features you're missing", ('triage',)),
    ('hey orion', ()),
    ('is it going to rain later', ()),
    ('skip this song', ()),
    ('wake me up at 7', ()),
    ("what's the date today", ()),
    ('close chrome', ()),
    ("you're awesome", ()),
    ("what's 64 divided by 8", ()),
    ('lower the volume a bit', ()),
    ('see you tomorrow', ()),
    ("what's my next meeting", ()),
    ('take a screenshot', ()),
]


class _OneHotEncoder:
    """Each request means the label whose word it contains: a deterministic
    stand-in for the sentence model, to test the decision rules."""

    def __init__(self):
        import numpy as np
        from orion_core import agent_router
        self.np = np
        self.labels = list(agent_router.EXEMPLARS)

    def ready(self):
        return True

    def _vector(self, text):
        vector = self.np.zeros(len(self.labels) + 1, dtype=self.np.float32)
        for index, label in enumerate(self.labels):
            if label and f"[{label}]" in text:
                vector[index] = 1.0
        if not vector.any():
            vector[self.labels.index("")] = 1.0
        return vector / self.np.linalg.norm(vector)

    def encode(self, texts, gpu=True):
        from orion_core import agent_router
        rows = []
        for label, items in agent_router.EXEMPLARS.items():
            for _ in items:
                rows.append(self._vector(f"[{label}]" if label else ""))
        return self.np.vstack(rows)

    def encode_one(self, text):
        return self._vector(text)


#: A third set, written 29 Oct before the second round of routing changes
#: (cues, more exemplars) and scored only in aggregate until they were done.
ROUTING_FRESH = [
    ('how do I get more people to sign up to my newsletter', ('marketing',)),
    ('write a product description for handmade candles that sells', ('marketing',)),
    ('should I run a discount or a giveaway to launch my store', ('marketing',)),
    ('my tiktok views dropped off a cliff this month', ('marketing',)),
    ("what's a good slogan for a dog walking business", ('marketing',)),
    ('IndexError: list index out of range on line 12', ('coding',)),
    ('make a python script that renames every photo by its date', ('coding',)),
    ('why does my css grid collapse on mobile', ('coding',)),
    ("refactor this class so it's easier to test", ('coding',)),
    ('my unit tests pass locally but fail on CI', ('coding',)),
    ('is it better to rent or buy a house right now in the uk', ('research',)),
    ('look up reviews of the three best budget laptops and compare them', ('research',)),
    ('what caused the 2008 financial crisis', ('research',)),
    ('put together a summary of the evidence on intermittent fasting', ('research', 'neuroscience')),
    ('what does the hippocampus do', ('neuroscience',)),
    ('why do we dream', ('neuroscience',)),
    ('explain dopamine and motivation for my exam', ('neuroscience',)),
    ('how does the brain recover after a stroke', ('neuroscience',)),
    ('what is a transformer in deep learning', ('aiml',)),
    ('my model overfits after five epochs', ('aiml',)),
    ('how do embeddings capture meaning', ('aiml',)),
    ('should I use random forest or gradient boosting for this dataset', ('aiml',)),
    ('someone texted me a link saying my parcel is held, is it real', ('security',)),
    ('what is a sql injection and how do I stop it', ('security', 'coding')),
    ('should I turn on two factor for my bank', ('security',)),
    ('my router has a default admin password, does that matter', ('security',)),
    ('you keep cutting out halfway through sentences', ('reliability',)),
    ('why did you freeze when I opened the globe', ('reliability',)),
    ('your replies are taking ages today', ('reliability',)),
    ('check your logs for any errors since this morning', ('reliability',)),
    ('hook up my google calendar so you can read it', ('integrations',)),
    ('the github api keeps returning 401 for me', ('integrations',)),
    ('can you post to a discord channel through a webhook', ('integrations',)),
    ('connect my philips hue lights to you', ('integrations',)),
    ('where did we get to with the essay yesterday', ('continuity',)),
    ('pick up where we left off on the robot project', ('continuity',)),
    ("what did I say I'd finish by friday", ('continuity',)),
    ("give me a quick summary of what we've done on this so far", ('continuity',)),
    ('my downloads folder is a mess, sort it out', ('files',)),
    ("find every pdf I've saved this year", ('files',)),
    ('delete duplicate photos on my drive', ('files',)),
    ('how much space are my videos taking up', ('files',)),
    ("what can't you do yet that you should", ('triage',)),
    ('list the things you got wrong this week', ('triage',)),
    ('which of your tools fail the most', ('triage',)),
    ('what would make you more useful to me', ('triage',)),
    ('good morning orion', ()),
    ("what's the forecast for tomorrow", ()),
    ('skip this track', ()),
    ('set a timer for twenty minutes', ()),
    ('open spotify', ()),
    ('how are you feeling today', ()),
    ("what's fifteen percent of eighty", ()),
    ('turn the volume up a bit', ()),
    ('read me the news', ()),
    ("thanks, that's all", ()),
    ('minimise all my windows', ()),
    ('what day is it', ()),
]


def test_meaning_decides_with_a_clear_lead_and_orion_keeps_small_talk(monkeypatch):
    from orion_core import agent_router
    manager = _manager()
    monkeypatch.setattr(agent_router, "ENABLED", True)
    encoder = _OneHotEncoder()
    agent_router._matrix.clear()
    agents = dict(manager._agents)

    def choose(request):
        agent, _conf, how = agent_router.choose(request, manager.route_with_confidence(request), agents,
                                                encoder, keyword_threshold=manager.ROUTE_THRESHOLD)
        return getattr(agent, "name", None), how

    assert choose("tidy my stuff [files]") == ("files", "meaning")
    assert choose("hello there") == (None, "none")                   # ORION answers himself
    assert choose("[files] and [security] at once")[1] in ("keywords", "none")   # no clear lead
    strong = "debug this python traceback in my code, refactor the function"
    assert choose(strong + " [files]") == ("coding", "keywords")      # strong keywords win
    agent_router._matrix.clear()

    class NotReady:
        def ready(self):
            return False

    picked = agent_router.choose(strong, manager.route_with_confidence(strong), agents, NotReady(),
                                 keyword_threshold=manager.ROUTE_THRESHOLD)
    assert getattr(picked[0], "name", None) == "coding" and picked[2] == "keywords"
    weak = "send an email"                                            # a lone keyword: below threshold
    assert agent_router.choose(weak, manager.route_with_confidence(weak), agents, NotReady(),
                               keyword_threshold=manager.ROUTE_THRESHOLD)[0] is None


def test_code_self_and_service_cues_lean_on_their_specialist():
    """Shapes that point at one specialist whatever the wording."""
    from orion_core import agent_router
    encoder = _OneHotEncoder()
    agent_router._matrix.clear()
    plain = agent_router.similarities("tell me about it", encoder)
    for request, label in (("IndexError: list index out of range", "coding"),
                           ("why are you using so much memory", "reliability"),
                           ("pull my history from the spotify api", "integrations")):
        cued = agent_router.similarities(request, encoder)
        assert cued[label] - plain[label] == pytest.approx(agent_router.CUES[label][1]), request
        others = {name for name in agent_router.CUES if name != label}
        assert all(cued[name] == plain[name] for name in others), request
    agent_router._matrix.clear()


def _routing_score(cases, route):
    right = small_talk_routed = 0
    for request, accept in cases:
        name = getattr(route(request), "name", None)
        right += (name in accept) if accept else (name is None)
        small_talk_routed += (not accept and name is not None)
    return right / len(cases), small_talk_routed


def test_the_real_sentence_model_routes_by_meaning(monkeypatch):
    """With ORION's own encoder (skipped where it is not installed)."""
    from orion_core import agent_router, semantic
    # A private encoder: loading the shared one changed later memory tests
    # (they expect it unloaded) - seen in the 29 Oct full run.
    monkeypatch.setattr(semantic, "ENCODER", semantic.SemanticEncoder())
    agent_router._matrix.clear()
    if not semantic.ENCODER.installed() or not semantic.ENCODER.warm(allow_download=False):
        pytest.skip("the sentence model is not installed here")
    monkeypatch.setattr(agent_router, "ENABLED", True)
    manager = _manager()
    seen, seen_small = _routing_score(ROUTING_SEEN, manager.route)
    unseen, unseen_small = _routing_score(ROUTING_UNSEEN, manager.route)
    fresh, fresh_small = _routing_score(ROUTING_FRESH, manager.route)
    # 29 Oct, second round: 93.2%, 94.0% and 89.7% (first round 93.2%, 80.6%, 82.8%).
    # Third round (the decision rule, encoder on the CPU): 96.1%, 94.0%, 93.1%.
    assert seen >= 0.93 and unseen >= 0.90 and fresh >= 0.88, (seen, unseen, fresh)
    assert seen_small == 0 and unseen_small == 0 and fresh_small <= 1
    monkeypatch.setattr(agent_router, "ENABLED", False)               # the keyword router, as before
    old_seen, _ = _routing_score(ROUTING_SEEN, manager.route)
    assert old_seen < 0.5


def test_study_work_and_requests_for_care_get_a_reviewing_pass():
    from orion_core.agents import wants_care
    for asked in ("double-check this for my exam", "explain LTP for my dissertation",
                  "be thorough about the proof", "help with my coursework essay"):
        assert wants_care(asked), asked
    for asked in ("what is ltp", "write a quick email", "open spotify"):
        assert not wants_care(asked), asked


def test_a_specialist_answers_on_the_local_model_when_the_cloud_fails_and_reviews_study_work():
    from types import SimpleNamespace

    calls = []

    class Router:
        def has_text_fallback(self):
            return True

        def local_text_profiles(self):
            return [SimpleNamespace(name="local_ollama")]

        async def generate_text(self, prompt, system_extra="", task=""):
            calls.append(("cloud", prompt[:20]))
            if "DRAFT ANSWER" in prompt:
                return SimpleNamespace(name="gemini"), "Reviewed: long-term potentiation strengthens synapses."
            raise RuntimeError("503 from the cloud")

        async def generate_text_offline(self, prompt, system_extra=""):
            calls.append(("local", prompt[:20]))
            return SimpleNamespace(name="local_ollama"), "LTP strengthens synapses after repeated firing."

    logged = []
    manager = _manager()
    agent = manager.get("neuroscience")
    agent.router = Router()
    agent.bus = SimpleNamespace(log=SimpleNamespace(emit=logged.append),
                                agent_activity=SimpleNamespace(emit=lambda *_a: None))
    quick = asyncio.run(agent.handle("what is long-term potentiation"))
    assert quick.ok and "via local_ollama" in quick.text and "LTP strengthens" in quick.text
    assert any("local model" in line for line in logged)
    calls.clear()
    careful = asyncio.run(agent.handle("explain long-term potentiation for my exam"))
    assert careful.ok and "Reviewed:" in careful.text
    assert [where for where, _ in calls] == ["cloud", "local", "cloud"]

    class NoLocal(Router):
        def local_text_profiles(self):
            return []

    agent.router = NoLocal()
    failed = asyncio.run(agent.handle("what is long-term potentiation"))
    assert not failed.ok


# ── the decision rule, 29 Oct 2026 (third round) ─────────────────────────────
#
# Most misses on SEEN were specialist requests left with ORION because two
# specialists stood close together ("Why does my React component render
# twice?": coding 0.607, reliability 0.598, ORION far behind). Measured with
# ORION's own encoder on the CPU: SEEN 91.3% -> 96.1%, UNSEEN 94.0% -> 94.0%,
# FRESH 89.7% -> 93.1%, small talk sent to a specialist 0/0/1 -> 0/0/0.

def test_close_specialists_well_ahead_of_orion_go_to_the_closer_one(monkeypatch):
    from orion_core import agent_router
    agents = dict(_manager()._agents)

    def pick(scores):
        monkeypatch.setattr(agent_router, "similarities", lambda _request, _encoder=None: dict(scores))
        return getattr(agent_router.choose("a request", (None, 0), agents)[0], "name", None)

    assert pick({"coding": 0.607, "reliability": 0.598, "": 0.50}) == "coding"
    assert pick({"coding": 0.65, "reliability": 0.65, "": 0.50}) is None        # an exact tie
    assert pick({"reliability": 0.704, "": 0.695, "triage": 0.63}) is None      # too close to ORION
    assert pick({"coding": 0.59, "reliability": 0.40, "": 0.40}) is None        # too weak
    assert pick({"": 0.80, "coding": 0.79}) is None                             # ORION's own


def test_a_bare_sum_is_orions_and_orion_by_name_is_you():
    from orion_core import agent_router
    sums, _bonus = agent_router.CUES[""]
    for asked in ("what's fifteen percent of eighty", "What's 15 percent of 240?",
                  "what's 64 divided by 8", "what is twelve times eight", "calculate 3.5 * 12",
                  "one hundred and twenty minus forty"):
        assert sums.search(asked), asked
    for asked in ("why is 0.1 + 0.2 not 0.3 in python", "what caused the 2008 financial crisis",
                  "set a timer for 10 minutes", "IndexError: list index out of range on line 12",
                  "how many GPUs do I need to train a small language model?"):
        assert not sums.search(asked), asked
    you, _bonus = agent_router.CUES["reliability"]
    assert you.search("ORION keeps crashing after startup, diagnose it")
    assert not you.search("my program keeps crashing after startup")
