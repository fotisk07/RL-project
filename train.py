import argparse

import gymnasium as gym
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import text_flappy_bird_gym
from tqdm import tqdm

import wandb
from project import agents
from project.utils import record_episode

### Agent Registry ###
AGENTS = {
    "mc": agents.MCAgent,
    "sarsa": agents.SarsaLambdaAgent,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on TextFlappyBird")
    # Env params
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--pipe-gap", type=int, default=4)
    # Agent params
    parser.add_argument("--agent", type=str, default="mc", choices=AGENTS.keys())
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.9, help="Lambda for Sarsa(λ) trace decay")
    parser.add_argument("--seed", type=int, default=12345)
    # Logging
    parser.add_argument("--run-name", type=str, default="train")
    parser.add_argument("--log-q-every", type=int, default=500)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def q_to_wandb_images(agent):
    Q = agent.Q  # (height, width, 2)
    V = np.max(Q, axis=2)
    policy = np.argmax(Q, axis=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(V, cmap="RdYlGn", aspect="auto")
    axes[0].set_title("State Value V(s) = max_a Q(s,a)")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    plt.colorbar(im0, ax=axes[0])

    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])
    axes[1].imshow(policy, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Greedy Policy (red=no flap, green=flap)")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")

    plt.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def make_agent(args, rng):
    cls = AGENTS[args.agent]
    kwargs = {
        "height": args.height,
        "width": args.width,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "rng": rng,
    }
    if args.agent == "sarsa":
        kwargs["lam"] = args.lam
    return cls(**kwargs)


def run_episode_mc(env, agent, max_steps):
    """Collect a full trajectory, then learn at the end (MC style)."""
    state, _ = env.reset()
    done = False
    trajectory = []
    total_reward = 0

    for step in range(max_steps):
        current_state = state
        action = agent.step(state)
        state, reward, done, _, info = env.step(action)
        trajectory.append((current_state, int(action), reward))
        total_reward += reward
        if done:
            break

    agent.learn(trajectory)
    return total_reward, step + 1


def run_episode_sarsa(env, agent, max_steps):
    """Update online at every step (Sarsa style)."""
    state, _ = env.reset()
    action = agent.step(state)
    done = False
    total_reward = 0
    agent.reset_traces()

    for step in range(max_steps):
        next_state, reward, done, _, info = env.step(action)
        next_action = agent.step(next_state)
        agent.update(state, action, reward, next_state, next_action, done)
        state, action = next_state, next_action
        total_reward += reward
        if done:
            break

    agent.decay_epsilon()
    agent.reset_traces()
    return total_reward, step + 1


EPISODE_FNS = {
    "mc": run_episode_mc,
    "sarsa": run_episode_sarsa,
}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    config = vars(args)

    if not args.no_wandb:
        wandb.init(project="flappy-rl", name=args.run_name, config=config)

    agent = make_agent(args, rng)
    env = gym.make(
        "TextFlappyBird-v0",
        height=args.height,
        width=args.width,
        pipe_gap=args.pipe_gap,
    )

    # Separate env for recording — doesn't interfere with training env
    record_env = gym.make(
        "TextFlappyBird-v0",
        height=args.height,
        width=args.width,
        pipe_gap=args.pipe_gap,
    )

    run_episode = EPISODE_FNS[args.agent]

    for ep in tqdm(range(args.episodes)):
        total_reward, ep_length = run_episode(env, agent, args.max_steps)

        if not args.no_wandb:
            log_dict = {
                "episode": ep,
                "total_reward": total_reward,
                "episode_length": ep_length,
                "epsilon": agent.epsilon,
                "mean_q": float(np.mean(agent.Q)),
                "max_q": float(np.max(agent.Q)),
            }

            if ep % args.log_q_every == 0 or ep == args.episodes - 1:
                log_dict["q_function"] = q_to_wandb_images(agent)

                # Record video at same frequency as Q plots
                frames, vid_reward = record_episode(record_env, agent, args.max_steps)
                if len(frames) > 0:
                    video_array = np.stack(frames).transpose(0, 3, 1, 2)  # (T, 3, H, W)
                    log_dict["video"] = wandb.Video(
                        video_array,
                        fps=2,
                        format="mp4",
                        caption=f"ep={ep} reward={vid_reward:.1f}",
                    )

            wandb.log(log_dict)

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
