# Local conversation recording

Recording is optional and starts off. Open Preferences, confirm that everyone
being recorded has agreed, and select **Start recording**. Each new session
needs fresh agreement. The HUD shows recording state and provides a **Stop
recording** control. Opening Preferences or restarting ORION never starts a
recording.

The recorder shares the existing 16 kHz microphone stream. It respects microphone
mute, pause, standby and push-to-talk. It can retain room conversation while
the normal wake-name or owner-response gate is closed. It does not open a second
microphone. Default record-only mode keeps room speech out of the Live service
and prevents ORION from answering it; typed controls remain available.

Audio is written to a WAV under the active config directory's
`conversation_recordings/<session ID>/`. The adjacent manifest reports actual
saved duration, dropped audio, skipped analysis and transcription errors. Sessions
are limited to two hours. Recording and analysis use separate bounded queues;
slow transcription can lose analysis while retaining audio, and overload can
drop audio blocks. The status reports both conditions.

Transcription uses the available local offline recogniser. It makes no model
download or cloud transcription request. Without a local recogniser, the audio
can still be saved and the transcript is reported unavailable.

Named speakers require a confident match to an existing consented voiceprint.
Distinct unknown voices may receive tentative session labels such as Speaker 1.
Anonymous comparison anchors remain in memory and are discarded when the session
finishes. Ambiguous speech stays unknown. Labels estimate a complete segment;
rapid turns and overlapping speech are not reliably separated. A recognised
speaker never grants permission to execute an action.

Use the `conversation_recording` tool with:

- `status` or `list` to inspect recording state or find saved session IDs;
- `recall`, `session_id` and an optional literal `query` to read saved words;
- `stop` to end capture and drain the writer;
- `forget` with an exact stopped session ID to move audio and transcripts to
  the Recycle Bin. A refusal preserves the files and reports the reason.

Long-running local analysis may remain finalising after capture stops. Its
status is retained, and a new session cannot overwrite it. Shutdown permanently
closes the capture service for that process. Source tests use synthetic audio
and temporary stores; physical microphone quality and real speaker accuracy
need separate acceptance.
