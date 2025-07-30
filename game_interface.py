#!/usr/bin/env python3
"""
Enhanced Game Interface for Liberian Entrepreneurship Environment
Provides a clean, responsive visual experience with character avatars and intuitive controls.
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
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_SIZE = 500
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50
CELL_SIZE = GRID_SIZE // 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (80, 80, 80)
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
SKIN_TONE = (255, 218, 185)
DARK_SKIN = (139, 69, 19)
BUTTON_HOVER = (200, 220, 255)
BUTTON_ACTIVE = (150, 200, 255)

class GameInterface:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Liberian Entrepreneurship - Character Adventure")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)
        
        # Game state
        self.env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
        self.obs, self.info = self.env.reset()
        self.total_reward = 0
        self.step_count = 0
        self.game_mode = "manual"
        
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
        """Create organized UI buttons."""
        buttons = {}
        
        # Control panel (right side)
        panel_x = WINDOW_WIDTH - 350
        panel_y = 200 
        
        # Title and controls
        buttons["AI Play"] = pygame.Rect(panel_x, panel_y, 150, 40)
        buttons["Reset"] = pygame.Rect(panel_x + 160, panel_y, 150, 40)
        
        # Action buttons (organized in 2 columns)
        action_names = ["Study", "Loan", "Buy", "Sell", "Research", "Service"]
        for i, name in enumerate(action_names):
            col = i % 2
            row = i // 2
            x = panel_x + col * 160
            y = panel_y + 60 + row * 50
            buttons[name] = pygame.Rect(x, y, 150, 40)
        
        return buttons
    
    def draw_grid(self):
        """Draw the main game grid."""
        # Draw grid background
        grid_rect = pygame.Rect(GRID_OFFSET_X, GRID_OFFSET_Y, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, WHITE, grid_rect)
        pygame.draw.rect(self.screen, BLACK, grid_rect, 2)
        
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
        """Draw all game locations with character sprites."""
        # Markets (Green) - Shopkeeper characters
        for market in self.env.markets:
            x = GRID_OFFSET_X + market[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + market[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_shopkeeper_sprite(x, y)
            
        # Schools (Blue) - Teacher characters
        for school in self.env.schools:
            x = GRID_OFFSET_X + school[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + school[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_teacher_sprite(x, y)
            
        # Banks (Yellow) - Banker characters
        for bank in self.env.banks:
            x = GRID_OFFSET_X + bank[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + bank[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_banker_sprite(x, y)
            
        # Suppliers (Orange) - Merchant characters
        for supplier in self.env.suppliers:
            x = GRID_OFFSET_X + supplier[1] * CELL_SIZE + CELL_SIZE // 2
            y = GRID_OFFSET_Y + supplier[0] * CELL_SIZE + CELL_SIZE // 2
            self.draw_merchant_sprite(x, y)
    
    def draw_shopkeeper_sprite(self, x, y):
        """Draw shopkeeper character sprite."""
        # Shopkeeper body
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y + 5), 8)
        pygame.draw.rect(self.screen, GREEN, (x - 6, y + 8, 12, 12))
        
        # Shopkeeper head
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y - 3), 6)
        
        # Eyes
        pygame.draw.circle(self.screen, BLACK, (x - 2, y - 4), 1)
        pygame.draw.circle(self.screen, BLACK, (x + 2, y - 4), 1)
        
        # Smile
        pygame.draw.arc(self.screen, BLACK, (x - 2, y - 2, 4, 3), 0, 3.14, 2)
        
        # Market sign
        pygame.draw.rect(self.screen, YELLOW, (x - 8, y - 15, 16, 8))
        text = self.font_small.render("$", True, BLACK)
        self.screen.blit(text, (x - 4, y - 13))
    
    def draw_teacher_sprite(self, x, y):
        """Draw teacher character sprite."""
        # Teacher body
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y + 5), 8)
        pygame.draw.rect(self.screen, BLUE, (x - 6, y + 8, 12, 12))
        
        # Teacher head
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y - 3), 6)
        
        # Glasses
        pygame.draw.circle(self.screen, BLACK, (x - 2, y - 4), 2, 1)
        pygame.draw.circle(self.screen, BLACK, (x + 2, y - 4), 2, 1)
        pygame.draw.line(self.screen, BLACK, (x, y - 4), (x + 1, y - 4), 1)
        
        # Book
        pygame.draw.rect(self.screen, WHITE, (x + 8, y - 2, 6, 8))
        pygame.draw.rect(self.screen, BLACK, (x + 8, y - 2, 6, 8), 1)
    
    def draw_banker_sprite(self, x, y):
        """Draw banker character sprite."""
        # Banker body
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y + 5), 8)
        pygame.draw.rect(self.screen, YELLOW, (x - 6, y + 8, 12, 12))
        
        # Banker head
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y - 3), 6)
        
        # Hat
        pygame.draw.rect(self.screen, BLACK, (x - 4, y - 8, 8, 4))
        
        # Eyes
        pygame.draw.circle(self.screen, BLACK, (x - 2, y - 4), 1)
        pygame.draw.circle(self.screen, BLACK, (x + 2, y - 4), 1)
        
        # Mustache
        pygame.draw.arc(self.screen, BLACK, (x - 3, y - 1, 6, 3), 3.14, 6.28, 2)
    
    def draw_merchant_sprite(self, x, y):
        """Draw merchant character sprite."""
        # Merchant body
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y + 5), 8)
        pygame.draw.rect(self.screen, ORANGE, (x - 6, y + 8, 12, 12))
        
        # Merchant head
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y - 3), 6)
        
        # Turban
        pygame.draw.circle(self.screen, PURPLE, (x, y - 6), 5)
        
        # Eyes
        pygame.draw.circle(self.screen, BLACK, (x - 2, y - 4), 1)
        pygame.draw.circle(self.screen, BLACK, (x + 2, y - 4), 1)
        
        # Beard
        pygame.draw.arc(self.screen, BLACK, (x - 3, y, 6, 4), 0, 3.14, 2)
        
        # Crate
        pygame.draw.rect(self.screen, BROWN, (x - 6, y - 6, 12, 8))
        pygame.draw.rect(self.screen, BLACK, (x - 6, y - 6, 12, 8), 1)
    
    def draw_agent(self):
        """Draw the player agent with character sprite."""
        x = GRID_OFFSET_X + self.env.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
        y = GRID_OFFSET_Y + self.env.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
        
        # Animated agent
        animation_offset = int(3 * np.sin(self.animation_frame * 0.2))
        
        # Agent body (student)
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y + 5 + animation_offset), 8)
        pygame.draw.rect(self.screen, RED, (x - 6, y + 8 + animation_offset, 12, 12))
        
        # Agent head
        pygame.draw.circle(self.screen, SKIN_TONE, (x, y - 3 + animation_offset), 6)
        
        # Hair
        pygame.draw.circle(self.screen, BLACK, (x, y - 6 + animation_offset), 4)
        
        # Eyes
        pygame.draw.circle(self.screen, WHITE, (x - 2, y - 4 + animation_offset), 2)
        pygame.draw.circle(self.screen, WHITE, (x + 2, y - 4 + animation_offset), 2)
        pygame.draw.circle(self.screen, BLACK, (x - 2, y - 4 + animation_offset), 1)
        pygame.draw.circle(self.screen, BLACK, (x + 2, y - 4 + animation_offset), 1)
        
        # Smile
        pygame.draw.arc(self.screen, BLACK, (x - 2, y - 1 + animation_offset, 4, 3), 0, 3.14, 2)
        
        # Backpack
        pygame.draw.rect(self.screen, BLUE, (x + 8, y + 2 + animation_offset, 6, 8))
        pygame.draw.rect(self.screen, BLACK, (x + 8, y + 2 + animation_offset, 6, 8), 1)
    
    def draw_ui(self):
        """Draw the organized user interface."""
        # Background panel
        panel_rect = pygame.Rect(WINDOW_WIDTH - 380, 0, 380, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Title
        title = self.font_large.render("Liberian Entrepreneurship", True, BLACK)
        self.screen.blit(title, (WINDOW_WIDTH - 370, 10))
        
        # Game stats
        stats_y = 50
        stats = [
            f"Money: ${self.info['money']:.1f}",
            f"Business Level: {self.info['business_level']:.1f}",
            f"Step: {self.step_count}",
            f"Reward: {self.total_reward:.1f}",
            f"Skills: {np.mean(list(self.info['skills'].values())):.2f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font_small.render(stat, True, BLACK)
            self.screen.blit(text, (WINDOW_WIDTH - 370, stats_y + i * 25))
        
        # Draw buttons
        self.draw_buttons()
        
        # Draw instructions and location guide
        self.draw_instructions()
        
        # Draw message
        if self.message and self.message_timer > 0:
            self.draw_message()
    
    def draw_buttons(self):
        """Draw all UI buttons with hover effects."""
        for name, rect in self.buttons.items():
            # Determine button color based on state
            if name == self.hovered_button:
                color = BUTTON_HOVER
            elif name == self.selected_action:
                color = BUTTON_ACTIVE
            else:
                color = WHITE
            
            # Draw button background
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            
            # Button text
            if name in ["Study", "Loan", "Buy", "Sell", "Research", "Service"]:
                text = self.font_small.render(name, True, BLACK)
            else:
                text = self.font_small.render(name, True, BLACK)
            
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_instructions(self):
        """Draw control instructions and location guide."""
        # Control instructions
        instructions = [
            "Controls:",
            "Arrow Keys: Move",
            "1-6: Actions",
            "7,8,9,0: Diagonal",
            "Mouse: Click action buttons"
        ]
        
        y_start = WINDOW_HEIGHT - 200
        for i, instruction in enumerate(instructions):
            text = self.font_small.render(instruction, True, BLACK)
            self.screen.blit(text, (WINDOW_WIDTH - 370, y_start + i * 20))
        
        # Location guide/key
        guide_title = self.font_small.render("📍 LOCATION GUIDE:", True, (50, 50, 150))
        self.screen.blit(guide_title, (WINDOW_WIDTH - 370, y_start + 120))
        
        guide_items = [
            "👤 You: Student entrepreneur",
            "🟢 Markets: Sell products for money",
            "🔵 Schools: Study to improve skills", 
            "🟡 Banks: Apply for loans",
            "🟠 Suppliers: Buy inventory"
        ]
        
        for i, item in enumerate(guide_items):
            text = self.font_small.render(item, True, BLACK)
            self.screen.blit(text, (WINDOW_WIDTH - 370, y_start + 140 + i * 18))
    
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
        if button_name in ["Study", "Loan", "Buy", "Sell", "Research", "Service"]:
            action_map = {"Study": 8, "Loan": 9, "Buy": 10, "Sell": 11, "Research": 12, "Service": 13}
            self.execute_action(action_map[button_name])
            
        elif button_name == "AI Play":
            self.start_ai_mode()
            
        elif button_name == "Reset":
            self.reset_game()
    
    def handle_key_press(self, key):
        """Handle keyboard input with arrow keys."""
        key_actions = {
            pygame.K_UP: 0,      # Up arrow
            pygame.K_RIGHT: 2,    # Right arrow
            pygame.K_DOWN: 4,     # Down arrow
            pygame.K_LEFT: 6,     # Left arrow
            pygame.K_1: 8,        # Study
            pygame.K_2: 9,        # Loan
            pygame.K_3: 10,       # Buy
            pygame.K_4: 11,       # Sell
            pygame.K_5: 12,       # Research
            pygame.K_6: 13,       # Service
            # Diagonal movement
            pygame.K_7: 7,        # Northwest
            pygame.K_8: 1,        # Northeast
            pygame.K_9: 3,        # Southeast
            pygame.K_0: 5,        # Southwest
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
        self.message_timer = 60
        
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
    print("Starting Enhanced Liberian Entrepreneurship Game...")
    print("Controls:")
    print("- Arrow Keys: Movement")
    print("- 1-6: Actions (Study, Loan, Buy, Sell, Research, Service)")
    print("- 7,8,9,0: Diagonal movement")
    print("- Mouse: Click buttons")
    print("- ESC: Quit")
    
    game = GameInterface()
    game.run()

if __name__ == "__main__":
    main() 