#!/usr/bin/env python3
"""
Atari-style Game Interface for Liberian Entrepreneurship Environment
Provides a visual, game-like experience with sprites, animations, and UI elements.
"""

import sys
import os
import pygame
import numpy as np
import time
from pygame import gfxdraw

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))
sys.path.append(os.path.join(PROJECT_ROOT, 'training'))

from environment.custom_env import LiberianEntrepreneurshipEnv
from training.dqn_training import DQNAgent
from training.pg_training import REINFORCEAgent

# Initialize Pygame
pygame.init()

# Game Constants
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
GRID_SIZE = 512
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
CELL_SIZE = GRID_SIZE // 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
DARK_GREEN = (0, 100, 0)
DARK_BLUE = (0, 0, 100)
DARK_YELLOW = (150, 150, 0)
DARK_ORANGE = (150, 100, 0)
BROWN = (139, 69, 19)
LIGHT_BLUE = (173, 216, 230)
LIGHT_GREEN = (144, 238, 144)

class GameInterface:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Liberian Entrepreneurship - Atari Style")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Game state
        self.env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.game_mode = "manual"  # manual, ai, demo
        
        # UI elements
        self.buttons = self.create_buttons()
        self.hovered_button = None
        self.selected_action = None
        self.message = ""
        self.message_timer = 0
        
        # Animation
        self.animation_frame = 0
        self.last_animation_time = time.time()
        
    def create_buttons(self):
        """Create UI buttons with better positioning."""
        buttons = {}
        
        # Action buttons (left side of panel)
        action_names = ["Study", "Loan", "Buy", "Sell", "Research", "Service"]
        for i, name in enumerate(action_names):
            x = WINDOW_WIDTH - 220
            y = 150 + i * 55
            buttons[name] = pygame.Rect(x, y, 180, 45)
            
        # Movement buttons (8-directional) - centered
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        center_x = WINDOW_WIDTH - 125
        center_y = 520
        radius = 70
        
        for i, direction in enumerate(directions):
            angle = i * 45 * np.pi / 180
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            buttons[direction] = pygame.Rect(x - 25, y - 25, 50, 50)
            
        # Control buttons
        buttons["AI Play"] = pygame.Rect(WINDOW_WIDTH - 220, 50, 180, 45)
        buttons["Reset"] = pygame.Rect(WINDOW_WIDTH - 220, 100, 180, 45)
        
        return buttons
    
    def draw_grid(self):
        """Draw the main game grid."""
        # Draw grid background
        grid_rect = pygame.Rect(GRID_OFFSET_X, GRID_OFFSET_Y, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, WHITE, grid_rect)
        pygame.draw.rect(self.screen, BLACK, grid_rect, 3)
        
        # Draw grid lines
        for i in range(11):
            x = GRID_OFFSET_X + i * CELL_SIZE
            y = GRID_OFFSET_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, LIGHT_GRAY, (x, GRID_OFFSET_Y), (x, GRID_OFFSET_Y + GRID_SIZE))
            pygame.draw.line(self.screen, LIGHT_GRAY, (GRID_OFFSET_X, y), (GRID_OFFSET_X + GRID_SIZE, y))
        
        # Draw locations
        self.draw_locations()
        
        # Draw agent
        self.draw_agent()
        
    def draw_locations(self):
        """Draw all game locations with sprites."""
        # Markets (Green)
        for market in self.env.markets:
            x = GRID_OFFSET_X + market[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + market[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_market_sprite(x, y)
            
        # Schools (Blue)
        for school in self.env.schools:
            x = GRID_OFFSET_X + school[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + school[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_school_sprite(x, y)
            
        # Banks (Yellow)
        for bank in self.env.banks:
            x = GRID_OFFSET_X + bank[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + bank[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_bank_sprite(x, y)
            
        # Suppliers (Orange)
        for supplier in self.env.suppliers:
            x = GRID_OFFSET_X + supplier[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + supplier[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_supplier_sprite(x, y)
    
    def draw_market_sprite(self, x, y):
        """Draw market building sprite."""
        # Building base
        pygame.draw.rect(self.screen, DARK_GREEN, (x - 15, y - 10, 30, 20))
        pygame.draw.rect(self.screen, GREEN, (x - 12, y - 8, 24, 16))
        
        # Market sign
        pygame.draw.rect(self.screen, YELLOW, (x - 8, y - 15, 16, 8))
        text = self.font_small.render("$", True, BLACK)
        self.screen.blit(text, (x - 4, y - 13))
        
        # Windows
        pygame.draw.rect(self.screen, WHITE, (x - 8, y - 2, 6, 6))
        pygame.draw.rect(self.screen, WHITE, (x + 2, y - 2, 6, 6))
    
    def draw_school_sprite(self, x, y):
        """Draw school building sprite."""
        # Building base
        pygame.draw.rect(self.screen, DARK_BLUE, (x - 15, y - 10, 30, 20))
        pygame.draw.rect(self.screen, BLUE, (x - 12, y - 8, 24, 16))
        
        # School flag
        pygame.draw.rect(self.screen, WHITE, (x + 8, y - 20, 4, 12))
        pygame.draw.rect(self.screen, RED, (x + 8, y - 20, 4, 6))
        
        # Door
        pygame.draw.rect(self.screen, BROWN, (x - 4, y + 2, 8, 10))
    
    def draw_bank_sprite(self, x, y):
        """Draw bank building sprite."""
        # Building base
        pygame.draw.rect(self.screen, DARK_YELLOW, (x - 15, y - 10, 30, 20))
        pygame.draw.rect(self.screen, YELLOW, (x - 12, y - 8, 24, 16))
        
        # Bank columns
        pygame.draw.rect(self.screen, GRAY, (x - 12, y - 8, 4, 16))
        pygame.draw.rect(self.screen, GRAY, (x + 8, y - 8, 4, 16))
        
        # Bank sign
        text = self.font_small.render("B", True, BLACK)
        self.screen.blit(text, (x - 4, y - 2))
    
    def draw_supplier_sprite(self, x, y):
        """Draw supplier building sprite."""
        # Building base
        pygame.draw.rect(self.screen, DARK_ORANGE, (x - 15, y - 10, 30, 20))
        pygame.draw.rect(self.screen, ORANGE, (x - 12, y - 8, 24, 16))
        
        # Warehouse doors
        pygame.draw.rect(self.screen, BROWN, (x - 10, y + 2, 8, 10))
        pygame.draw.rect(self.screen, BROWN, (x + 2, y + 2, 8, 10))
        
        # Crate
        pygame.draw.rect(self.screen, BROWN, (x - 6, y - 6, 12, 8))
    
    def draw_agent(self):
        """Draw the player agent with animation."""
        x = GRID_OFFSET_X + self.env.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
        y = GRID_OFFSET_Y + self.env.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
        
        # Animated agent
        animation_offset = int(5 * np.sin(self.animation_frame * 0.2))
        
        # Agent body
        pygame.draw.circle(self.screen, RED, (x, y + animation_offset), 12)
        pygame.draw.circle(self.screen, (200, 0, 0), (x, y + animation_offset), 8)
        
        # Agent eyes
        pygame.draw.circle(self.screen, WHITE, (x - 4, y + animation_offset - 2), 3)
        pygame.draw.circle(self.screen, WHITE, (x + 4, y + animation_offset - 2), 3)
        pygame.draw.circle(self.screen, BLACK, (x - 4, y + animation_offset - 2), 1)
        pygame.draw.circle(self.screen, BLACK, (x + 4, y + animation_offset - 2), 1)
        
        # Agent direction indicator
        if self.selected_action is not None and self.selected_action < 8:
            angle = self.selected_action * 45 * np.pi / 180
            end_x = x + 15 * np.cos(angle)
            end_y = y + 15 * np.sin(angle)
            pygame.draw.line(self.screen, WHITE, (x, y), (end_x, end_y), 3)
    
    def draw_ui(self):
        """Draw the user interface."""
        # Background panel
        panel_rect = pygame.Rect(WINDOW_WIDTH - 280, 0, 280, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Title
        title = self.font_large.render("Liberian", True, BLACK)
        subtitle = self.font_large.render("Entrepreneurship", True, BLACK)
        self.screen.blit(title, (WINDOW_WIDTH - 270, 10))
        self.screen.blit(subtitle, (WINDOW_WIDTH - 270, 50))
        
        # Game stats
        stats_y = 100
        stats = [
            f"Money: ${self.info['money']:.1f}",
            f"Business Level: {self.info['business_level']:.1f}",
            f"Step: {self.step_count}",
            f"Reward: {self.total_reward:.1f}",
            f"Skills: {np.mean(list(self.info['skills'].values())):.2f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font_small.render(stat, True, BLACK)
            self.screen.blit(text, (WINDOW_WIDTH - 270, stats_y + i * 25))
        
        # Draw buttons
        self.draw_buttons()
        
        # Draw message
        if self.message and self.message_timer > 0:
            self.draw_message()
    
    def draw_buttons(self):
        """Draw all UI buttons with hover effects."""
        for name, rect in self.buttons.items():
            # Determine button color based on state
            if name == self.hovered_button:
                color = LIGHT_GREEN if name in ["Study", "Loan", "Buy", "Sell", "Research", "Service"] else LIGHT_BLUE
            elif name == self.selected_action:
                color = GREEN
            else:
                color = LIGHT_GRAY
            
            # Draw button background
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            
            # Button text
            if name in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
                text = self.font_small.render(name, True, BLACK)
            else:
                text = self.font_small.render(name, True, BLACK)
            
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_message(self):
        """Draw status message."""
        if self.message:
            text = self.font_medium.render(self.message, True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, 50))
            
            # Message background
            bg_rect = text_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, YELLOW, bg_rect)
            pygame.draw.rect(self.screen, BLACK, bg_rect, 2)
            
            self.screen.blit(text, text_rect)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.handle_mouse_click(event.pos)
                    
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event.pos)
                
            elif event.type == pygame.KEYDOWN:
                self.handle_key_press(event.key)
        
        return True
    
    def handle_mouse_motion(self, pos):
        """Handle mouse motion for hover effects."""
        self.hovered_button = None
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                self.hovered_button = name
                break
    
    def handle_mouse_click(self, pos):
        """Handle mouse clicks on buttons."""
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                self.handle_button_click(name)
                break
    
    def handle_button_click(self, button_name):
        """Handle button clicks."""
        print(f"Button clicked: {button_name}")  # Debug print
        
        if button_name in ["Study", "Loan", "Buy", "Sell", "Research", "Service"]:
            action_map = {"Study": 8, "Loan": 9, "Buy": 10, "Sell": 11, "Research": 12, "Service": 13}
            self.execute_action(action_map[button_name])
            
        elif button_name in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            action_map = {"N": 0, "NE": 1, "E": 2, "SE": 3, "S": 4, "SW": 5, "W": 6, "NW": 7}
            self.execute_action(action_map[button_name])
            
        elif button_name == "AI Play":
            self.start_ai_mode()
            
        elif button_name == "Reset":
            self.reset_game()
    
    def handle_key_press(self, key):
        """Handle keyboard input."""
        key_actions = {
            pygame.K_w: 0,  # North
            pygame.K_e: 1,  # NE
            pygame.K_d: 2,  # East
            pygame.K_c: 3,  # SE
            pygame.K_s: 4,  # South
            pygame.K_z: 5,  # SW
            pygame.K_a: 6,  # West
            pygame.K_q: 7,  # NW
            pygame.K_1: 8,  # Study
            pygame.K_2: 9,  # Loan
            pygame.K_3: 10, # Buy
            pygame.K_4: 11, # Sell
            pygame.K_5: 12, # Research
            pygame.K_6: 13, # Service
        }
        
        if key in key_actions:
            self.execute_action(key_actions[key])
    
    def execute_action(self, action):
        """Execute a game action."""
        self.selected_action = action
        
        # Take step in environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        
        # Update message
        action_names = ["North", "NE", "East", "SE", "South", "SW", "West", "NW",
                       "Study", "Loan", "Buy", "Sell", "Research", "Service"]
        self.message = f"{action_names[action]}: {reward:.1f}"
        self.message_timer = 60  # Show message for 60 frames
        
        # Check game end
        if terminated:
            if self.info['money'] <= 0:
                self.message = "❌ BANKRUPTCY! Game Over!"
            elif self.info['business_level'] >= 3 and self.info['money'] >= 500:
                self.message = "✅ SUCCESS! You built a successful business!"
            else:
                self.message = "⏰ Time's up!"
            self.message_timer = 120
    
    def start_ai_mode(self):
        """Start AI playing mode."""
        self.game_mode = "ai"
        self.message = "🤖 AI is playing..."
        self.message_timer = 60
    
    def reset_game(self):
        """Reset the game."""
        self.env.close()
        self.env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.selected_action = None
        self.message = "🔄 Game Reset!"
        self.message_timer = 60
    
    def update_animation(self):
        """Update animation frame."""
        self.animation_frame += 1
        if self.message_timer > 0:
            self.message_timer -= 1
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update animation
            self.update_animation()
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Draw game elements
            self.draw_grid()
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        self.env.close()
        pygame.quit()

def main():
    """Main function."""
    print("Starting Atari-style Liberian Entrepreneurship Game...")
    print("Controls:")
    print("- WASD/QE/ZC: Movement")
    print("- 1-6: Actions (Study, Loan, Buy, Sell, Research, Service)")
    print("- Mouse: Click buttons")
    print("- ESC: Quit")
    
    game = GameInterface()
    game.run()

if __name__ == "__main__":
    main() 