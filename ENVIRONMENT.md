# 💀 PettingZoo `battle_v4` Explained (Jujutsu Kaisen Themed)

---

## 🪨 Setting: Domain Expansion - The Battlefield

**PettingZoo `battle_v4`** simulates a cursed battlefield where:

- Two teams (Red vs Blue) are locked in a **Domain Expansion**-level fight.
- Each agent (sorcerer or cursed spirit) fights individually to survive and defeat enemies.
- The arena is a giant cursed gridworld where survival, attacks, and positioning determine who wins.

---

## 🪩 Teams

- ✨ Red Team: Jujutsu Sorcerers
- ⚫ Blue Team: Cursed Spirits

Agents are **split evenly** between teams at the start.
Each agent is independent and controls its own fate within the Domain.

---

## 🌐 Arena Layout

- Open 2D grid (like Shibuya city blocks)
- No major obstacles: pure open field chaos
- Agents spawn near their side but can move anywhere

> “The battlefield itself becomes a lethal cursed territory.”

---

## 🔫 Agent Actions

| Action | Meaning |
|:---|:---|
| 0 | Move North |
| 1 | Move South |
| 2 | Move East |
| 3 | Move West |
| 4 | Stay still (charge cursed energy) |
| 5 | Attack forward (release cursed energy beam) |

**Facing direction matters** — attacks only hit forward!

---

## ❤️ Health and Damage System

- Agents have a small HP pool (like Cursed Spirit cores)
- Attacks deal fixed damage
- When HP drops to zero: **"Exorcised!"** or **"Defeated!"**

---

## 👁️ Observation (What Each Sorcerer/Spirit Sees)

Each agent receives a **local Spirit Radar view**:

- Size: **13x13x5 tensor**

| Channel | Meaning |
|:---|:---|
| 0 | Presence of self |
| 1 | Presence of teammates |
| 2 | Presence of enemies |
| 3 | Health of visible agents (normalized) |
| 4 | Attack cooldown info |

> “Just like sensing cursed energy — you only feel what's near you.”

**NO full global knowledge!**
Each agent must survive with local cursed senses only.

---

## ✨ Reward Structure

| Event | Reward |
|:---|:---|
| Successful attack | +1 |
| Killing an enemy | +5 |
| Surviving longer | Indirect reward (more chances to attack) |

**Higher cursed output = greater reward!**

---

## ⌛ Episode Ending

- Team completely defeated
- OR maximum steps reached (exhausted cursed energy)

> “Victory or exorcism — no in-between.”

---

# 🌟 Jujutsu Kaisen Spirit Summary

| Thing | Meaning |
|:---|:---|
| Agents | Sorcerers and Cursed Spirits |
| Grid Arena | Shibuya-like battlefield |
| Actions | Movement, attacks, cursed defense |
| Observations | Spirit Energy Radar (partial vision) |
| Rewards | Based on survival and exorcism |

✅ Tactical mastery over the Domain required for true victory.

---

# 💥 TL;DR

- `battle_v4` = massive chaotic Domain Expansion
- Agents sense nearby cursed energy (partial observations)
- Move, attack, survive, dominate
- Build true cursed technique mastery over episodes

**Train to control the Domain. Conquer to survive.** 🌊👁️

---

# 📈 Bonus

**Once trained, agents will:**
- Flank enemies
- Flee from overpowering threats
- Target weakened opponents
- Dominate with cursed precision

Just like true Jujutsu Kaisen masters. 🚀