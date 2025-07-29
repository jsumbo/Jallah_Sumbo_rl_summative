import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import random
from typing import Optional, Tuple, Dict, Any

class LiberianEntrepreneurshipEnv(gym.Env):
    """
    Custom Gymnasium environment for Liberian Entrepreneurship Simulation.
    
    The environment simulates a young Liberian student learning entrepreneurial skills
    through decision-making scenarios. The agent must manage resources, make market
    decisions, and develop skills to succeed as an entrepreneur.
    
    Environment Components:
    - Agent: A Liberian secondary school student
    - Goal: Develop entrepreneurial skills and build a successful business
    - Challenges: Resource constraints, market volatility, skill development
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment parameters
        self.grid_size = 10
        self.max_steps = 200
        self.current_step = 0
        
        # Agent state components
        self.agent_pos = np.array([0, 0])  # Position in the grid
        self.money = 100.0  # Starting capital
        self.skills = {
            'business_planning': 0.1,
            'market_analysis': 0.1, 
            'financial_management': 0.1,
            'leadership': 0.1,
            'innovation': 0.1
        }
        self.business_level = 0  # 0: Idea, 1: Startup, 2: Growing, 3: Established
        self.market_knowledge = 0.0
        self.customer_satisfaction = 0.5
        self.inventory = 0
        
        # Market conditions (dynamic)
        self.market_demand = 0.5
        self.competition_level = 0.3
        self.economic_stability = 0.7
        
        # Grid elements
        self.markets = [(2, 3), (7, 6), (5, 8)]  # Market locations
        self.schools = [(1, 1), (8, 2)]  # Skill development locations
        self.banks = [(4, 4), (6, 9)]  # Financing locations
        self.suppliers = [(3, 7), (9, 5)]  # Inventory locations
        
        # Action space: 8 movement directions + 6 interaction actions
        # 0-7: Move (N, NE, E, SE, S, SW, W, NW)
        # 8: Study/Learn (at school)
        # 9: Apply for loan (at bank)
        # 10: Buy inventory (at supplier)
        # 11: Sell products (at market)
        # 12: Market research
        # 13: Improve customer service
        self.action_space = spaces.Discrete(14)
        
        # Observation space: normalized values
        # [agent_x, agent_y, money, business_level, market_demand, 
        #  competition_level, economic_stability, market_knowledge,
        #  customer_satisfaction, inventory, skill_avg]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 512
        
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        skill_avg = np.mean(list(self.skills.values()))
        
        obs = np.array([
            self.agent_pos[0] / (self.grid_size - 1),  # Normalized x position
            self.agent_pos[1] / (self.grid_size - 1),  # Normalized y position
            min(self.money / 1000.0, 1.0),  # Normalized money (capped at 1000)
            self.business_level / 3.0,  # Normalized business level
            self.market_demand,  # Market demand
            self.competition_level,  # Competition level
            self.economic_stability,  # Economic stability
            self.market_knowledge,  # Market knowledge
            self.customer_satisfaction,  # Customer satisfaction
            min(self.inventory / 100.0, 1.0),  # Normalized inventory
            skill_avg  # Average skill level
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "money": self.money,
            "skills": self.skills.copy(),
            "business_level": self.business_level,
            "market_conditions": {
                "demand": self.market_demand,
                "competition": self.competition_level,
                "stability": self.economic_stability
            },
            "step": self.current_step
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset agent state
        self.agent_pos = np.array([0, 0])
        self.money = 100.0
        self.skills = {
            'business_planning': 0.1,
            'market_analysis': 0.1,
            'financial_management': 0.1,
            'leadership': 0.1,
            'innovation': 0.1
        }
        self.business_level = 0
        self.market_knowledge = 0.0
        self.customer_satisfaction = 0.5
        self.inventory = 0
        self.current_step = 0
        
        # Reset market conditions with some randomness
        self.market_demand = 0.3 + random.random() * 0.4  # 0.3 to 0.7
        self.competition_level = 0.2 + random.random() * 0.4  # 0.2 to 0.6
        self.economic_stability = 0.5 + random.random() * 0.4  # 0.5 to 0.9
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        
        # Movement actions (0-7)
        if action < 8:
            reward += self._move_agent(action)
        
        # Interaction actions (8-13)
        elif action == 8:  # Study/Learn
            reward += self._study_action()
        elif action == 9:  # Apply for loan
            reward += self._loan_action()
        elif action == 10:  # Buy inventory
            reward += self._buy_inventory_action()
        elif action == 11:  # Sell products
            reward += self._sell_products_action()
        elif action == 12:  # Market research
            reward += self._market_research_action()
        elif action == 13:  # Improve customer service
            reward += self._improve_service_action()
        
        # Update market conditions (dynamic environment)
        self._update_market_conditions()
        
        # Check termination conditions
        if self.money <= 0:
            terminated = True
            reward -= 50  # Penalty for bankruptcy
        elif self.business_level >= 3 and self.money >= 500:
            terminated = True
            reward += 100  # Bonus for successful business
        elif self.current_step >= self.max_steps:
            truncated = True
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, truncated, info
    
    def _move_agent(self, action: int) -> float:
        """Move the agent in the specified direction."""
        # Direction mappings: N, NE, E, SE, S, SW, W, NW
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        
        dx, dy = directions[action]
        new_pos = self.agent_pos + np.array([dx, dy])
        
        # Check bounds
        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.agent_pos = new_pos
            return 0.1  # Small reward for valid movement
        
        return -0.5  # Penalty for invalid movement
    
    def _study_action(self) -> float:
        """Study/learn action - only effective at schools."""
        if tuple(self.agent_pos) in self.schools:
            if self.money >= 10:
                self.money -= 10
                # Randomly improve a skill
                skill_to_improve = random.choice(list(self.skills.keys()))
                self.skills[skill_to_improve] = min(1.0, self.skills[skill_to_improve] + 0.1)
                return 5.0  # Good reward for skill development
            else:
                return -1.0  # Penalty for insufficient funds
        return -2.0  # Penalty for studying at wrong location
    
    def _loan_action(self) -> float:
        """Apply for loan - only effective at banks."""
        if tuple(self.agent_pos) in self.banks:
            # Loan approval based on business level and financial management skill
            approval_chance = (self.business_level * 0.3 + 
                             self.skills['financial_management'] * 0.7)
            
            if random.random() < approval_chance:
                loan_amount = 50 + (self.business_level * 25)
                self.money += loan_amount
                return 3.0  # Reward for successful loan
            else:
                return -1.0  # Penalty for loan rejection
        return -2.0  # Penalty for applying at wrong location
    
    def _buy_inventory_action(self) -> float:
        """Buy inventory - only effective at suppliers."""
        if tuple(self.agent_pos) in self.suppliers:
            cost_per_unit = 5.0
            max_units = int(self.money / cost_per_unit)
            
            if max_units > 0:
                units_to_buy = min(max_units, 10)  # Buy up to 10 units
                total_cost = units_to_buy * cost_per_unit
                self.money -= total_cost
                self.inventory += units_to_buy
                return 2.0  # Reward for inventory purchase
            else:
                return -1.0  # Penalty for insufficient funds
        return -2.0  # Penalty for buying at wrong location
    
    def _sell_products_action(self) -> float:
        """Sell products - only effective at markets."""
        if tuple(self.agent_pos) in self.markets:
            if self.inventory > 0:
                # Sales success based on market conditions and skills
                sales_multiplier = (
                    self.market_demand * 0.4 +
                    (1 - self.competition_level) * 0.3 +
                    self.skills['market_analysis'] * 0.3
                )
                
                units_sold = min(self.inventory, int(10 * sales_multiplier) + 1)
                revenue_per_unit = 8.0 * (1 + self.customer_satisfaction * 0.5)
                total_revenue = units_sold * revenue_per_unit
                
                self.money += total_revenue
                self.inventory -= units_sold
                
                # Improve business level based on sales performance
                if units_sold >= 5 and self.business_level < 3:
                    self.business_level += 0.1
                
                return 10.0 * (units_sold / 10.0)  # Reward proportional to sales
            else:
                return -2.0  # Penalty for no inventory
        return -2.0  # Penalty for selling at wrong location
    
    def _market_research_action(self) -> float:
        """Conduct market research."""
        if self.money >= 5:
            self.money -= 5
            self.market_knowledge = min(1.0, self.market_knowledge + 0.2)
            # Update market demand knowledge
            self.market_demand = min(1.0, self.market_demand + 0.1)
            return 3.0  # Reward for market research
        return -1.0  # Penalty for insufficient funds
    
    def _improve_service_action(self) -> float:
        """Improve customer service."""
        if self.money >= 8:
            self.money -= 8
            self.customer_satisfaction = min(1.0, self.customer_satisfaction + 0.15)
            return 4.0  # Reward for service improvement
        return -1.0  # Penalty for insufficient funds
    
    def _update_market_conditions(self):
        """Update market conditions dynamically."""
        # Small random changes to market conditions
        self.market_demand += random.uniform(-0.05, 0.05)
        self.market_demand = np.clip(self.market_demand, 0.1, 1.0)
        
        self.competition_level += random.uniform(-0.03, 0.03)
        self.competition_level = np.clip(self.competition_level, 0.1, 0.8)
        
        self.economic_stability += random.uniform(-0.02, 0.02)
        self.economic_stability = np.clip(self.economic_stability, 0.3, 1.0)
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        
        pix_square_size = self.window_size / self.grid_size
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        
        # Draw locations
        # Markets (green)
        for market in self.markets:
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * market[1],
                    pix_square_size * market[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Schools (blue)
        for school in self.schools:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    pix_square_size * school[1],
                    pix_square_size * school[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Banks (yellow)
        for bank in self.banks:
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                pygame.Rect(
                    pix_square_size * bank[1],
                    pix_square_size * bank[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Suppliers (orange)
        for supplier in self.suppliers:
            pygame.draw.rect(
                canvas,
                (255, 165, 0),
                pygame.Rect(
                    pix_square_size * supplier[1],
                    pix_square_size * supplier[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw agent (red circle)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                int(pix_square_size * (self.agent_pos[1] + 0.5)),
                int(pix_square_size * (self.agent_pos[0] + 0.5)),
            ),
            int(pix_square_size / 3),
        )
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

