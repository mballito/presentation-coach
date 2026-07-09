# Changelog

## v1.3 — 2026-07-09
### Added
- 🎓 **The Network Mentor** — technical accuracy scoring for Learning/CCNA scenarios
- Technical metrics: Accuracy, Clarity, Teaching Quality, Depth, Analogies, Structure
- Technically relevant feedback instead of generic soft-skills tips

## v1.2 — 2026-07-09
### Added
- 📸 **Dress & Presentation photo analysis** (gemma3:4b vision)
- ✏️ **Custom scenario input** — paste job URL or write your own description
- 🎙️ **Live Conversations placeholder category**
- **5 new categories**: Negotiation (8), Networking (7), Tech Interviews (9), Career Growth (8), Client Management (9)
- **20 extra Learning scenarios** covering ACLs, EtherChannel, IPv6, HSRP, Port Security, QoS, STP enhancements, WAN/VPNs, Syslog/SNMP
- **Whisper model preloaded at startup** — faster response, no OOM
- **Global error handler** with demo-mode fallback

### Fixed
- VAD silence threshold: 40th → 15th percentile (clamped at 30% for near-silence)
- Pitch variety thresholds lowered for conversational speech
- Category-aware feedback (no empathy tips for tech/learning topics)
- Category-aware coach assignment (1 coach per category)

## v1.1 — 2026-07-09
### Added
- **Learning — Teach It** category (10 CCNA scenarios with teaching tips)
- GitHub backup (private repo mballito/presentation-coach)
- Total: 109 scenarios across 12 categories

## v1.0 — 2026-07-09
### Added
- Initial MVP with 3 coaches (Comforter, Closer, Igniter)
- 7 categories, 79 scenarios
- Real audio analysis (pitch, volume, pauses, pace)
- Gamification (points, levels, leaderboard)
- Dark theme with glassmorphism UI
