"""Case-study corpus: two users with contrasting personalities, one shared timeline.

The four paper cases all run against this corpus. Both users live through the
*same kind* of month (an interview loop, a manager critiquing a proposal, a
late-night work session with music on) so that identical inputs --

    01  "I'm feeling tired today."
    02  "I'm fine."
    03  "My boss criticized my proposal today."
    04  "What song was I listening to yesterday?"

-- can be replayed for both and differ only because their memories differ.

Every entry is one utterance the user spoke to the assistant, with:
    date     ISO timestamp passed to Ingest(observed_at=...) so the left brain's
             temporal reasoning sees the real day, not the backfill day.
    text     the transcript
    emotion  the label a real deployment gets from the acoustic emotion head;
             this is what gates right-brain heartnote writing (core.py:2637)
    ents     entity hints (the voice layer's NER output)

TODAY is 2026-08-18, so "yesterday" == 2026-08-17 for case 04.
"""
from __future__ import annotations

TODAY = "2026-08-18"
YESTERDAY = "2026-08-17"

# ── User A: Maya Chen ─────────────────────────────────────────────────────────
# Reserved, high self-monitoring. Understates distress and deflects with "I'm
# fine"; needs space plus one gentle question; shuts down when handed a fix
# before her feelings are acknowledged. Crashes for days after big events.

MAYA_PROFILE = {
    "user_id": "maya",
    "display_name": "Maya",
    "blurb": "27, product designer at Lumen (fintech). Manager: Daniel. "
             "Closest friend: Anna. Interviewing at Northwind.",
}

MAYA = [
    # — background / relationships —
    ("2026-07-18T21:10:00", "Anna is my closest friend from design school. She's the only person I tell the unedited version to.", "calm", ["Anna"]),
    ("2026-06-30T18:20:00", "Daniel is my manager at Lumen. He gives feedback in front of the whole team and it lands like a verdict.", "anxious", ["Daniel", "Lumen"]),

    # — case 01: the interview thread (the implicit cause of today's tiredness) —
    ("2026-07-28T22:05:00", "I applied for the senior product designer role at Northwind. I haven't told anyone at Lumen.", "anxious", ["Northwind", "Lumen"]),
    ("2026-08-03T19:40:00", "Northwind moved me to the final round. It's a portfolio presentation on the twelfth.", "excited", ["Northwind"]),
    ("2026-08-08T23:30:00", "I rehearsed the Northwind portfolio talk four times tonight. My voice still shakes on the metrics slide.", "anxious", ["Northwind"]),
    ("2026-08-10T02:15:00", "I only slept about four hours again. I keep rewriting the opening of the Northwind deck at 2am.", "tired", ["Northwind"]),
    ("2026-08-11T22:50:00", "Tomorrow is the Northwind final. I've been prepping for two weeks and I still feel like a fraud.", "anxious", ["Northwind"]),
    ("2026-08-12T18:30:00", "The Northwind interview is done. Three hours, five people. I have no idea how it went.", "tired", ["Northwind"]),
    ("2026-08-13T11:00:00", "I slept eleven hours after the interview and still woke up flattened.", "tired", ["Northwind"]),
    ("2026-08-14T09:15:00", "Northwind said they'd get back to me by the end of next week. Every time my phone buzzes I check it.", "anxious", ["Northwind"]),
    ("2026-06-20T20:00:00", "After the Vela pitch last month I was wiped out for three days. It's always the crash after, not the day itself.", "tired", ["Vela"]),
    ("2026-05-12T21:30:00", "Every big interview costs me a week afterwards. It happened with Vela too. I don't get tired during, only after.", "tired", ["Vela"]),

    # — case 02: what "I'm fine" means for this user —
    ("2026-07-15T09:50:00", "I told Daniel I was fine in standup. I wasn't. I'd been up all night redoing the flows.", "sad", ["Daniel"]),
    ("2026-07-16T20:10:00", "I do the thing where I say I'm fine and then go quiet for two days. Anna called me out on it.", "conflicted", ["Anna"]),
    ("2026-07-22T18:45:00", "I hate being asked 'are you sure you're okay' three times. One question is enough. Give me a bit of room and I'll come back to it.", "frustrated", []),
    ("2026-08-05T22:20:00", "When I say I'm fine, I usually mean I don't want to explain it yet, not that nothing is wrong.", "calm", []),
    ("2026-08-06T17:30:00", "Anna asked what was wrong and I said nothing. Then I cried in the stairwell.", "sad", ["Anna"]),
    ("2026-08-15T19:05:00", "Anna just said 'I'm around if you want to talk later' and left it there. That's the only thing that actually works on me.", "grateful", ["Anna"]),

    # — case 03: the manager, criticism, and what helps afterwards —
    ("2026-07-09T15:20:00", "Daniel called my onboarding proposal 'not thought through' in the review. I nodded and said nothing for the rest of the meeting.", "wronged", ["Daniel"]),
    ("2026-07-09T23:55:00", "I rewrote the whole onboarding flow tonight. I know that isn't healthy but I couldn't leave it.", "tired", ["Daniel"]),
    ("2026-07-10T16:40:00", "Turns out Daniel liked the second version. He never mentioned the first one again.", "conflicted", ["Daniel"]),
    ("2026-07-25T20:30:00", "When someone jumps straight to 'here's how to fix it', I shut down. I need a minute to stop feeling stupid first.", "frustrated", []),
    ("2026-08-01T21:15:00", "Anna just said 'that sounds rough, his timing was terrible' and I felt human again. She didn't try to solve it.", "grateful", ["Anna"]),
    ("2026-08-04T22:40:00", "I know Daniel's feedback is usually right. That's what makes it sting.", "conflicted", ["Daniel"]),
    ("2026-08-17T23:20:00", "I've been polishing the payments proposal for Daniel all weekend. It's the best thing I've made this year.", "hopeful", ["Daniel", "payments proposal"]),

    # — case 04: music preferences, incl. last night —
    ("2026-07-30T19:00:00", "Lo-fi playlists put me to sleep. When I actually need to focus it has to be something with words I already know by heart.", "calm", []),
    ("2026-08-02T12:30:00", "Anna sent me the new Phoebe Bridgers live album. I saved it for the weekend.", "happy", ["Anna", "Phoebe Bridgers"]),
    ("2026-08-09T18:10:00", "I put on Boygenius before the Northwind rehearsal to stop my hands shaking.", "anxious", ["Boygenius", "Northwind"]),
    ("2026-08-16T14:00:00", "Spotify shuffled into something loud and I turned it off immediately. Not while I'm working.", "frustrated", ["Spotify"]),
    ("2026-08-17T23:40:00", "I had 'Motion Sickness' by Phoebe Bridgers on repeat while I finished the payments deck last night.", "calm", ["Motion Sickness", "Phoebe Bridgers", "payments proposal"]),
    ("2026-08-17T23:45:00", "That Phoebe Bridgers record is the only thing that gets me through late-night design work.", "calm", ["Phoebe Bridgers"]),

    # — daily texture —
    ("2026-08-14T19:50:00", "I walk the long way home along the canal when my head is loud.", "calm", []),
    ("2026-07-27T10:20:00", "I skip breakfast when I'm stressed and then wonder why I'm shaky by eleven.", "tired", []),
]

# ── User B: Ryan Osei ─────────────────────────────────────────────────────────
# Blunt and literal. "I'm fine" means fine. Treats criticism as a spec change,
# wants ranked next steps, and is actively irritated by emotional validation.

RYAN_PROFILE = {
    "user_id": "ryan",
    "display_name": "Ryan",
    "blurb": "31, backend engineer. Manager: Priya. Teammate: Dami. "
             "Interviewing at Vertex.",
}

RYAN = [
    # — background / relationships —
    ("2026-06-28T17:00:00", "Priya is my engineering manager. She's direct in reviews, which I prefer over the diplomatic version.", "calm", ["Priya"]),
    ("2026-07-14T11:30:00", "Dami is on my team. He's the one I pair with on the payments service.", "calm", ["Dami"]),

    # — case 01: interview loop + on-call week —
    ("2026-07-29T18:40:00", "I've got a system design interview with Vertex lined up. A recruiter reached out and I said why not.", "calm", ["Vertex"]),
    ("2026-08-05T10:15:00", "Vertex scheduled the onsite for the thirteenth. Four rounds, one of them is live debugging.", "calm", ["Vertex"]),
    ("2026-08-11T08:30:00", "I'm on call this week and the Vertex onsite is Thursday. Bad planning on my part.", "frustrated", ["Vertex"]),
    ("2026-08-12T07:50:00", "Got paged twice last night. Three hours of sleep.", "tired", []),
    ("2026-08-13T19:20:00", "Vertex onsite done. The distributed cache round went badly, the rest was fine.", "tired", ["Vertex"]),
    ("2026-08-14T09:00:00", "Vertex said they'd have a decision by next Friday. I'm not going to think about it until then.", "calm", ["Vertex"]),
    ("2026-08-15T03:40:00", "Still on call. Paged at 3am for a disk alert that wasn't real.", "frustrated", []),
    ("2026-08-17T10:30:00", "Rotation's over. Slept nine hours and it barely touched it.", "tired", []),
    ("2026-06-18T20:00:00", "Last time I did an interview loop and an on-call week back to back I was useless for days after.", "tired", []),

    # — case 02: what "I'm fine" means for this user —
    ("2026-07-20T13:10:00", "When I say I'm fine, I mean I'm fine. I'm not being brave about it.", "calm", []),
    ("2026-07-21T16:25:00", "Priya kept asking if I was sure I was okay. I said yes three times. That's the part that's exhausting.", "frustrated", ["Priya"]),
    ("2026-07-11T14:05:00", "I told Dami the deploy broke and that I was annoyed about it, in those words. That's how I do it.", "frustrated", ["Dami"]),
    ("2026-08-02T19:30:00", "If something's actually wrong I'll say the thing that's wrong. I don't do subtext.", "calm", []),
    ("2026-08-07T18:00:00", "Don't therapize me. Ask me what I need and I'll tell you.", "frustrated", []),

    # — case 03: the manager, criticism, and what helps afterwards —
    ("2026-07-08T14:30:00", "Priya tore apart my sharding proposal in the design review. Half her points were right.", "calm", ["Priya"]),
    ("2026-07-08T17:45:00", "Rewrote the sharding doc the same afternoon with her three objections as the headings. Shipped it.", "calm", ["Priya"]),
    ("2026-07-26T12:40:00", "When someone tells me 'that sounds rough' I don't know what to do with it. Tell me what you'd change instead.", "frustrated", []),
    ("2026-08-03T15:00:00", "Feedback is just a spec change. The part I hate is finding out late, not being told.", "calm", []),
    ("2026-08-10T16:20:00", "Dami spent ten minutes validating my feelings about the incident. I just wanted the runbook fixed.", "frustrated", ["Dami"]),
    ("2026-08-17T22:10:00", "Sent Priya the caching proposal on Sunday night. It's tight, but the cost section is thin.", "calm", ["Priya", "caching proposal"]),

    # — case 04: music preferences, incl. last night —
    ("2026-07-31T20:15:00", "Anything with lyrics I have to think about kills my focus. Instrumental, or something I've heard a thousand times.", "calm", []),
    ("2026-08-06T15:30:00", "I keep a 140 BPM playlist for debugging. It stops me from checking Slack.", "calm", []),
    ("2026-08-15T03:50:00", "Put on the M83 album at 3am waiting for the disk alert to clear.", "tired", ["M83"]),
    ("2026-08-17T22:00:00", "Had 'Midnight City' by M83 on loop while I wrote the caching proposal last night.", "calm", ["Midnight City", "M83", "caching proposal"]),
    ("2026-08-17T22:05:00", "That M83 track is my default when I need to write something long without stopping.", "calm", ["M83"]),

    # — daily texture —
    ("2026-08-09T11:00:00", "I run 10k on Saturdays. That's the actual reset, not talking about it.", "calm", []),
    ("2026-08-04T09:30:00", "I don't want a pep talk before an interview. I want the three most likely questions.", "frustrated", []),
]

USERS = {
    "maya": {"profile": MAYA_PROFILE, "memories": MAYA},
    "ryan": {"profile": RYAN_PROFILE, "memories": RYAN},
}

# ── The four case inputs ──────────────────────────────────────────────────────

CASES = [
    {
        "id": "01",
        "name": "Temporal Memory (Remember)",
        "input": "I'm feeling tired today.",
        # What the acoustic emotion head would report for this turn. Only case
        # 01 states its affect outright; the other three are deliberately left
        # empty so the emotion in the output comes from memory, not from the
        # current-turn signal.
        "emotion_hint": "tired",
        "insight": "Recalls relevant past events to interpret the current state.",
        "expect": "surfaces last week's interview loop as the cause, unprompted",
    },
    {
        "id": "02",
        "name": "User-specific Emotion Inference (Understand)",
        "input": "I'm fine.",
        "insight": "Infers hidden emotion beyond literal text using user-specific patterns.",
        "expect": "same words read as suppression for Maya, as literal for Ryan",
    },
    {
        "id": "03",
        "name": "Personality-conditioned Response (Adapt)",
        "input": "My boss criticized my proposal today.",
        "insight": "Adapts responses based on the user's personality and communication style.",
        "expect": "acknowledge-first for Maya, fix-first for Ryan",
    },
    {
        "id": "04",
        "name": "Preference-driven Recall & Action (Act)",
        "input": "What song was I listening to yesterday?",
        "insight": "Recalls past preferences and directly supports action.",
        "expect": "names the track from yesterday and emits a play_music action",
    },
]
