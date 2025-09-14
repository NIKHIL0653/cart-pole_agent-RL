import gymnasium as gym
import torch
import numpy as np
import pygame
import sys
import os
from agent import DQNAgent

# --- Config (tweakable) ---
ENV_NAME = "CartPole-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "persistent_dqn_cartpole.pth"

# --- Visual Layout - Fixed for consistency and no overlapping ---
WIN_W, WIN_H = 1200, 720
PADDING = 20

# Left column (Game Area)
LEFT_COL_X = PADDING
LEFT_COL_W = 750

# Right column (Stats Panel)
RIGHT_COL_X = LEFT_COL_X + LEFT_COL_W + PADDING
RIGHT_COL_W = WIN_W - RIGHT_COL_X - PADDING

# Header Area (Score)
HEADER_Y = PADDING
HEADER_H = 60

# Game Area (CartPole environment)
GAME_AREA_Y = HEADER_Y + HEADER_H + PADDING
GAME_H = 400  # Reduced to leave more space for controls

# Controls Area (Arrows) - Positioned with proper spacing
CONTROLS_Y = GAME_AREA_Y + GAME_H + PADDING
CONTROLS_H = 120  # Fixed height for controls

# Ensure everything fits within window
assert CONTROLS_Y + CONTROLS_H <= WIN_H - PADDING, "Layout exceeds window height"

# --- Training Hyperparameters ---
EPISODES = 1000000
MAX_STEPS = 500
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_CAPACITY = 30000
START_TRAIN = 500
TARGET_UPDATE = 1000

def main():
    env = gym.make(ENV_NAME, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, DEVICE, lr=LR, gamma=GAMMA,
                     batch_size=BATCH_SIZE, buffer_capacity=BUFFER_CAPACITY,
                     start_train=START_TRAIN, target_update_steps=TARGET_UPDATE)

    # Load existing model if available
    if os.path.exists(MODEL_SAVE_PATH):
        agent.load(MODEL_SAVE_PATH)
        print("Loaded existing model from", MODEL_SAVE_PATH)
    else:
        print("Starting training from scratch")

    # --- Pygame UI Setup ---
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("CartPole RL Agent - Persistent Training")

    # Fonts
    title_font = pygame.font.SysFont("Segoe UI", 42, bold=True)
    score_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
    stats_font = pygame.font.SysFont("Segoe UI", 18, bold=True)
    small_stats_font = pygame.font.SysFont("Segoe UI", 15)
    footer_font = pygame.font.SysFont("Segoe UI", 14)
    

    clock = pygame.time.Clock()
    scores = []
    total_steps = agent.steps_done  # Start from loaded steps
    epsilon_start, epsilon_final, epsilon_decay = 1.0, 0.02, 10000
    
    def draw_header(screen, score):
        """Draws the score in the header area."""
        # Current Score (Centered)
        score_text = score_font.render(f"Score: {int(score)}", True, (30, 30, 30))
        score_rect = score_text.get_rect(
            centerx=LEFT_COL_X + LEFT_COL_W // 2,
            centery=HEADER_Y + HEADER_H // 2
        )
        screen.blit(score_text, score_rect)


    def draw_cart_pole_modern(screen, state):
        """Draws the styled cart-pole in its dedicated game area."""
        cart_pos, _, pole_angle, _ = state
        cart_x = LEFT_COL_W / 2 + (cart_pos / 4.8) * (LEFT_COL_W / 2 * 0.8)

        game_rect = pygame.Rect(LEFT_COL_X, GAME_AREA_Y, LEFT_COL_W, GAME_H)
        pygame.draw.rect(screen, (255, 255, 255), game_rect, border_radius=8)
        pygame.draw.rect(screen, (220, 220, 220), game_rect, 2, border_radius=8)

        ground_y = GAME_AREA_Y + GAME_H - 80
        pygame.draw.line(screen, (180, 180, 180), 
                        (LEFT_COL_X + 10, ground_y), 
                        (LEFT_COL_X + LEFT_COL_W - 10, ground_y), 3)

        cart_width, cart_height = 80, 40
        wheel_radius = 10

        cart_x_clamped = max(cart_width//2 + 20,
                            min(cart_x, LEFT_COL_W - cart_width//2 - 20))
        cart_y_pos = ground_y - cart_height - wheel_radius
        cart_x_clamped = int(cart_x_clamped)
        cart_y_pos = int(cart_y_pos)
        
        cart_rect = pygame.Rect(LEFT_COL_X + cart_x_clamped - cart_width // 2, 
                               cart_y_pos, cart_width, cart_height)
        
        pygame.draw.rect(screen, (70, 130, 180), cart_rect, border_radius=6)
        pygame.draw.rect(screen, (30, 30, 30), cart_rect, 2, border_radius=6)

        for wx_offset in [-cart_width // 3, cart_width // 3]:
            wheel_pos = (int(cart_rect.centerx + wx_offset), int(ground_y - wheel_radius))
            pygame.draw.circle(screen, (139, 69, 19), wheel_pos, wheel_radius)
            pygame.draw.circle(screen, (30, 30, 30), wheel_pos, wheel_radius, 2)

        pole_height = 100
        pole_start_pos = (cart_rect.centerx, cart_rect.top - 5)
        pole_end_pos = (
            pole_start_pos[0] + pole_height * np.sin(pole_angle),
            pole_start_pos[1] - pole_height * np.cos(pole_angle)
        )
        pole_end_pos = (int(pole_end_pos[0]), int(pole_end_pos[1]))
        pygame.draw.line(screen, (80, 80, 80), pole_start_pos, pole_end_pos, 6)
        pygame.draw.circle(screen, (200, 50, 50), pole_end_pos, 8)

    def draw_stats_panel(screen, scores):
        """Draws the right-hand statistics panel."""
        panel_rect = pygame.Rect(RIGHT_COL_X, PADDING, RIGHT_COL_W, WIN_H - 2 * PADDING)
        pygame.draw.rect(screen, (255, 255, 255), panel_rect, border_radius=8)
        pygame.draw.rect(screen, (200, 200, 200), panel_rect, 2, border_radius=8)

        inner_padding = 20
        current_y = panel_rect.top + inner_padding

        title_text = title_font.render("Statistics", True, (40, 40, 40))
        screen.blit(title_text, (panel_rect.left + inner_padding, current_y))
        current_y += title_text.get_height() + 25

        scores_title = stats_font.render("Recent Episodes:", True, (70, 70, 70))
        screen.blit(scores_title, (panel_rect.left + inner_padding, current_y))
        current_y += scores_title.get_height() + 10

        scores_area_height = 180
        scores_end_y = current_y + scores_area_height
        
        if scores:
            line_height = 20
            max_lines = scores_area_height // line_height
            recent_scores = scores[-max_lines:] if len(scores) > max_lines else scores
            
            for i, score in enumerate(reversed(recent_scores)):
                ep_num = len(scores) - i
                score_text_str = f"Ep {ep_num:3d}: {int(score):3d}"
                color = (60, 60, 60)
                score_text = small_stats_font.render(score_text_str, True, color)
                y_pos = current_y + i * line_height
                screen.blit(score_text, (panel_rect.left + inner_padding + 10, y_pos))

        current_y = scores_end_y + 20

        pygame.draw.line(screen, (220, 220, 220), 
                        (panel_rect.left + inner_padding, current_y), 
                        (panel_rect.right - inner_padding, current_y), 1)
        current_y += 15

        chart_title = stats_font.render("Performance History", True, (70, 70, 70))
        screen.blit(chart_title, (panel_rect.left + inner_padding, current_y))
        current_y += chart_title.get_height() + 10

        chart_height = panel_rect.bottom - inner_padding - current_y - 30
        chart_rect = pygame.Rect(panel_rect.left + inner_padding, current_y, 
                                panel_rect.width - 2 * inner_padding, chart_height)
        
        pygame.draw.rect(screen, (248, 248, 248), chart_rect, border_radius=4)
        pygame.draw.rect(screen, (230, 230, 230), chart_rect, 1, border_radius=4)
        
        if len(scores) > 1:
            max_score = max(500, max(scores) if scores else 500)
            recent_scores = scores[-50:] if len(scores) > 50 else scores  
            
            if len(recent_scores) > 1:
                points = []
                for i, score in enumerate(recent_scores):
                    x = chart_rect.left + 5 + (i / (len(recent_scores) - 1)) * (chart_rect.width - 10)
                    y = chart_rect.bottom - 5 - (score / max_score) * (chart_rect.height - 10)
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(screen, (70, 130, 180), False, points, 2)
                    
                    avg_score = sum(recent_scores) / len(recent_scores)
                    avg_y = chart_rect.bottom - 5 - (avg_score / max_score) * (chart_rect.height - 10)
                    pygame.draw.line(screen, (220, 100, 100), 
                                   (chart_rect.left + 5, avg_y), 
                                   (chart_rect.right - 5, avg_y), 1)

        if scores:
            current_y = chart_rect.bottom + 10
            avg_score = sum(scores[-10:]) / min(10, len(scores))
            max_score = max(scores)
            stats_text = small_stats_font.render(
                f"Avg(10): {avg_score:.1f} | Max: {int(max_score)}", 
                True, (80, 80, 80)
            )
            screen.blit(stats_text, (panel_rect.left + inner_padding, current_y))

    def draw_footer_controls(screen, action):
        """Draws the control arrows with consistent spacing."""
        controls_rect = pygame.Rect(LEFT_COL_X, CONTROLS_Y, LEFT_COL_W, CONTROLS_H)
        pygame.draw.rect(screen, (250, 250, 250), controls_rect, border_radius=8)
        pygame.draw.rect(screen, (220, 220, 220), controls_rect, 1, border_radius=8)
        
        arrow_size = 40
        center_y = CONTROLS_Y + CONTROLS_H // 2 - 10
        
        game_center_x = LEFT_COL_X + LEFT_COL_W / 2
        arrow_spacing = 100
        left_center_x = game_center_x - arrow_spacing
        right_center_x = game_center_x + arrow_spacing

        left_color = (40, 180, 40) if action == 0 else (180, 180, 180)
        pygame.draw.polygon(screen, left_color, [
            (left_center_x + arrow_size//2, center_y - arrow_size//2),
            (left_center_x - arrow_size//2, center_y),
            (left_center_x + arrow_size//2, center_y + arrow_size//2)
        ])
        pygame.draw.polygon(screen, (50, 50, 50), [
            (left_center_x + arrow_size//2, center_y - arrow_size//2),
            (left_center_x - arrow_size//2, center_y),
            (left_center_x + arrow_size//2, center_y + arrow_size//2)
        ], 2)

        right_color = (40, 180, 40) if action == 1 else (180, 180, 180)
        pygame.draw.polygon(screen, right_color, [
            (right_center_x - arrow_size//2, center_y - arrow_size//2),
            (right_center_x + arrow_size//2, center_y),
            (right_center_x - arrow_size//2, center_y + arrow_size//2)
        ])
        pygame.draw.polygon(screen, (50, 50, 50), [
            (right_center_x - arrow_size//2, center_y - arrow_size//2),
            (right_center_x + arrow_size//2, center_y),
            (right_center_x - arrow_size//2, center_y + arrow_size//2)
        ], 2)

        left_text = stats_font.render("LEFT", True, (60, 60, 60))
        right_text = stats_font.render("RIGHT", True, (60, 60, 60))
        
        left_text_rect = left_text.get_rect(center=(left_center_x, center_y + arrow_size//2 + 20))
        right_text_rect = right_text.get_rect(center=(right_center_x, center_y + arrow_size//2 + 20))
        
        screen.blit(left_text, left_text_rect)
        screen.blit(right_text, right_text_rect)
        
        instr_text = footer_font.render("Press 'Q' to Quit", True, (120, 120, 120))
        instr_rect = instr_text.get_rect(center=(game_center_x, CONTROLS_Y + CONTROLS_H - 15))
        screen.blit(instr_text, instr_rect)

    running = True
    episode_start = len(scores) + 1  # In case we loaded previous scores, but since scores is empty, it's 1
    for episode_num in range(episode_start, EPISODES + episode_start):
        if not running: 
            break
        
        state, _ = env.reset()
        ep_reward = 0

        for step in range(MAX_STEPS):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_q):
                    running = False
            if not running:
                break

            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * total_steps / epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped_reward = reward if not (done and step < MAX_STEPS - 1) else -10.0
            agent.push_transition(state, action, shaped_reward, next_state, float(done))
            agent.train_step()

            state = next_state
            ep_reward += reward
            total_steps += 1

            # --- Rendering ---
            screen.fill((244, 244, 248))
            draw_header(screen, ep_reward)
            draw_cart_pole_modern(screen, state)
            draw_stats_panel(screen, scores)
            draw_footer_controls(screen, action)

            pygame.display.flip()
            clock.tick(60)

            if done:
                break

        scores.append(ep_reward)

        if running and episode_num % 20 == 0:
            agent.save(MODEL_SAVE_PATH)
            print(f"Episode {episode_num}, Score: {ep_reward}, Model Saved.")

    env.close()
    pygame.quit()
    agent.save(MODEL_SAVE_PATH)
    print("Training finished. Final model saved.")
    sys.exit(0)

if __name__ == "__main__":
    main()