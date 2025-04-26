# About
This is my attempt to create a multi-agent trainer in the PettingZoo Arena environment. Have fun exploring!

# Understanding the (13, 13, 5) Observation Shape in MAGent `battle_v4`

---

## 🛍 Why is the observation shape **(13, 13, 5)**?

### Short Answer
Each agent in MAGent's `battle_v4` environment observes a **13×13** grid centered around itself, where **each cell** has **5 different features**.

---

## 🧬 Full Detailed Breakdown

- **`13×13` Grid:**
  - An agent doesn't see the entire map.
  - It sees a local **square window** around itself — a "vision range".
  - The vision extends **6 squares outward** in each direction: (2×6) + 1 (center) = 13.

- **`5 Channels (Features)` per Cell:**
  - Each grid cell has 5 different data layers ("features"), such as:
    - Presence of **friendly agents**
    - Presence of **enemy agents**
    - Presence of **obstacles**
    - **Attack range** indicators
    - Possibly **health**, **agent type**, or other states

Thus, for every agent:
- **Width** = 13 squares left/right
- **Height** = 13 squares up/down
- **Depth (channels)** = 5 features per tile

---

## 📄 Quick Analogy

| Normal Image            | MAGent Observation          |
| :---------------------- | :--------------------------- |
| 13×13 pixels             | 13×13 map squares          |
| 3 channels (RGB colors)  | 5 channels (battle features) |

You can think of the observation like a small image, except instead of Red-Green-Blue color channels, you have battle-related feature channels!

---

## 💡 Why Flatten the Observation?

When training a policy network like an MLP (multi-layer perceptron), you need a **1D vector** input.

- 13×13×5 = **845 total features**
- So, we **flatten** the (13,13,5) grid into a **single vector of 845 values**.

This flattened vector is then fed into the first Linear layer of the ActorCritic network.

---

## 📊 Visualization Idea (Optional Bonus)

You can actually plot the 13×13x5 observations to "see" what an agent sees!

Each feature channel can be shown as a heatmap, where:
- Bright squares might show enemies
- Dim squares show friendly units
- Black squares mean empty space

**This is super helpful for debugging!**

---

## 🏳️‍☠️ TL;DR

| Dimension | Meaning                          |
| :-------- | :------------------------------- |
| 13        | visible width (left-right vision) |
| 13        | visible height (up-down vision)   |
| 5         | features per tile (friend, enemy, obstacles, etc.) |

Always **flatten** (13×13×5) into 845 values before passing into an MLP!

---

## 🔄 Common Fixes for Code

- Use `np.prod(env.observation_space(agent).shape)` to compute input size.
- Flatten observations before feeding them into the network.
- Flatten again when storing into buffers.

---

Stay sharp out there on the battlefield, captain! 💪🌊

