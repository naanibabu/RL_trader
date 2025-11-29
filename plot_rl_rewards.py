import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='either "train" or "test"')
args = parser.parse_args()

a = np.load(f'rl_trader_rewards/{args.mode}.npy')

print(f"average reward: {a.mean():.2f}, min: {a.min():.2f}, max: {a.max():.2f}")

if args.mode == 'train':
  # show the training progress with better styling
  plt.figure(figsize=(12, 6))
  plt.plot(a, linewidth=2, alpha=0.7, label='Episode Reward')
  
  # Add moving average to see trend
  moving_avg = np.convolve(a, np.ones(20)/20, mode='valid')
  plt.plot(moving_avg, linewidth=2, color='red', label='20-Episode Moving Average')
  
  plt.xlabel('Episode', fontsize=12)
  plt.ylabel('Reward', fontsize=12)
  plt.title(f'Training Rewards - {args.mode}', fontsize=14, fontweight='bold')
  plt.grid(True, alpha=0.3)
  plt.legend(fontsize=10)
  plt.tight_layout()
  plt.savefig(f'rl_trader_rewards/training_rewards.png', dpi=300, bbox_inches='tight')
  
else:
  # test - show a histogram of rewards
  plt.figure(figsize=(10, 6))
  plt.hist(a, bins=20, edgecolor='black', alpha=0.7)
  plt.xlabel('Reward', fontsize=12)
  plt.ylabel('Frequency', fontsize=12)
  plt.title(f'Test Rewards Distribution - {args.mode}', fontsize=14, fontweight='bold')
  plt.grid(True, alpha=0.3, axis='y')
  plt.tight_layout()
  plt.savefig(f'rl_trader_rewards/test_rewards.png', dpi=300, bbox_inches='tight')

plt.show()