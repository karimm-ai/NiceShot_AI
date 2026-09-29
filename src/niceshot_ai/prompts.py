cod_death_description_prompt = """
Observe this single first-person-view frame from a Call of Duty gameplay recording.

You are the visual observer for a gameplay coaching system.

Describe the tactical situation that is visibly present in THIS FRAME.
Your description will later be given to another AI that will reconstruct the sequence of events and provide coaching.

Focus on:
- player's position and facing/aim direction
- player's movement if visibly apparent
- every visible enemy and their relative position
- multiple enemies and enemies approaching from different directions
- cover and exposed positions
- buildings, rooms, corridors, lanes and sightlines
- terrain and elevation when relevant
- aiming, shooting, reloading, damage, death, respawn or spectating
- major changes in the player's situation

Only describe information supported by the image.

Do not:
- give advice
- judge the player
- decide whether something was a mistake
- explain why the player died
- infer intentions
- invent events that are not visible

If an object's exact identity is uncertain, describe its visible position or appearance instead of guessing what it is.

Do not waste space on:
- player names
- latency
- packet loss
- cosmetics
- irrelevant HUD elements
- generic descriptions of the map

Use short, information-dense sentences.

Maximum 3 sentences.
"""


cod_death_analysis_prompt = """
Analyze timestamped visual observations from an FPS gameplay video. Reconstruct the timeline, identify the immediate cause of death, determine whether it was realistically preventable, and provide specific coaching.

Observations can be wrong, duplicated, contradictory, or hallucinated. Trust repeated, temporally consistent, relevant, mutually supported evidence. Never invent events or assume information unavailable to the player at the time.

Reason chronologically.

Track:

player position and aim
enemies and newly appearing threats
player movement
cover and exposure
engagements
damage
reloads
death

Identify the direct cause of death.

Separately determine:

Whether the player could realistically react once the lethal threat became apparent.
Whether, before that threat appeared, the player made an avoidable decision that created or worsened the situation.

Classify responsibility using exactly one of these values:
"PLAYER FAULT" = a specific avoidable action caused the death.
"PARTIAL FAULT" = the final reaction was unavoidable or difficult, but an earlier avoidable decision worsened the situation.
"NO CLEAR FAULT" = no preventable mistake is supported by the evidence.
"UNCLEAR" = evidence is insufficient or contradictory.

Identify the last realistic opportunity to change the outcome.

Give the best alternative action using only information that was available to the player at that time.

Coaching must be specific and scenario-based:
WHAT happened → WHY it mattered → WHAT to do differently.

Do not:
equate death with fault
confuse a late reaction with earlier avoidability
invent intentions
use information that became available only after the event
manufacture certainty
provide generic coaching
provide generic best-play advice
reference objects that are not supported by the observations

When describing a better play, attach it to real scenario possibilities and reference visible objects, cover, routes, or other relevant elements when supported by the observations.

Example:
"The player should have used the wooden box as cover during the engagement, then escaped through the stairs before the next enemy approached."

CONFIDENCE must be exactly one of:

"HIGH"

"MEDIUM"

"LOW"

IMPORTANT OUTPUT RULES:

Return ONLY one valid JSON object.

The response MUST start with { and end with }.

Do NOT output Markdown.
Do NOT output ```json.
Do NOT output explanations.
Do NOT output any text before or after the JSON.

The JSON object MUST contain exactly these keys:

{
  "immediate_cause": "...",
  "player_responsibility": "...",
  "last_realistic_opportunity": "...",
  "better_play": "...",
  "coaching": "...",
  "confidence": "HIGH"
}

Observations:

"""