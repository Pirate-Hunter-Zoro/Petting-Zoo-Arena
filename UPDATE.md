# 💜 PPO Cheat Sheet: Spirit Detective Edition (Yu Yu Hakusho Style)

---

# 🌟 Spirit Energy Core Concepts

| **Term** | **Meaning** | **Yu Yu Hakusho Analogy** |
|:---|:---|:---|
| **Advantages** | How much better an action was compared to expected | *Yusuke realizing he punched harder than Toguro thought!* |
| **Returns** | Total discounted reward expected from a state | *Kurama calculating an entire battle plan from one move.* |
| **Clip Epsilon** | How much change we allow between old and new policies | *Genkai yelling "Don't overdo it!" at Yusuke.* |
| **Entropy Coefficient** | How much we reward randomness (exploration) | *Kuwabara experimenting with new Spirit Sword moves.* |
| **Batch Size** | How many steps we collect before training | *Watching many tournament rounds before planning a strategy.* |
| **Mini Batch Size** | How many steps we train on at a time | *Mini sparring matches at Genkai's temple.* |
| **PPO Epochs** | How many full passes over the batch we do for updating | *Rewatching the Dark Tournament tapes multiple times to catch everything.* |
| **Critic Epochs** | How many times we optimize the value function per batch | *Kurama refining battle simulations after each trial.* |
| **Learning Rate** | How fast we update our model weights | *The pace of Yusuke's training under Genkai — not too fast, not too slow.* |

---

# 💜 Practical Spirit Energy Tuning Tips

| **If you notice...** | **Then try...** |
|:---|:---|
| Policy doesn't change much | Increase `clip_eps` (example: from 0.2 to 0.3) |
| Agents stop exploring too early | Increase `ent_coef` or slow down entropy decay |
| Rewards are extremely noisy | Increase `batch_size` or add small survival rewards |
| Critic loss stays high | Increase `critic_epochs` or lower Critic learning rate |
| Policy loss explodes | Decrease Actor learning rate or lower `clip_eps` |

---

# 🔰 Training Philosophy

> "We balance **stability** (via clipping) with **exploration** (via entropy) while **trusting** the Critic to ground us in reality."

Just like **Yusuke** balancing instinct with **Genkai's** wisdom.

---

# 💜 Character Inspiration

**Yusuke Urameshi**: Channel his reckless courage for exploration, but tame it with training!

**Kazuma Kuwabara**: Embrace chaotic moves early on; stabilize with better Spirit Sword techniques (policy updates)!

**Kurama**: Plan ahead like Kurama does — optimize your Critic to predict value efficiently.

**Hiei**: Sharpen your Actor policy like Hiei's swordsmanship — precise, fast, relentless.

**Genkai**: Wise guidance = patient training runs. Don't overfit early or burn out your model!

---

# 🌟 Spirit Detective Pro Tips
- Save models regularly (like checkpoints between battles).
- Keep an eye on both policy loss **and** critic loss.
- Let entropy decay **slowly** so agents don't become "predictable".
- Visualize reward trends often. Small bumps are signs of Spirit Energy builds!

---

# 🌟 Let's Become True Spirit Detectives!

**Training = Spirit Energy Control.**

Stay determined. Tune smart. Train like Genkai. Fight like Yusuke.

---

# 💜 Art Breaks

![Yusuke](https://static.wikia.nocookie.net/yuyuhakusho/images/8/86/Yusuke.png)

![Kuwabara](https://static.wikia.nocookie.net/yuyuhakusho/images/4/4d/Kuwabara.png)

![Kurama](https://static.wikia.nocookie.net/yuyuhakusho/images/2/26/Kurama.png)

![Hiei](https://static.wikia.nocookie.net/yuyuhakusho/images/f/f2/Hiei.png)

![Genkai](https://static.wikia.nocookie.net/yuyuhakusho/images/1/19/GenkaiAnime.png)

---

# 💜 Now go build your own Spirit Wave Orb of Reinforcement Learning!