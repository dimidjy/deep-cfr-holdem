import pygame
import numpy as np
from typing import Dict, List, Optional, Any
from src.engine import PokerEngine


class PokerGUI:
    
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    
    BACKGROUND_COLOR = (53, 101, 77)
    TABLE_COLOR = (35, 70, 50)
    CARD_COLOR = (255, 255, 255)
    TEXT_COLOR = (255, 255, 255)
    BUTTON_COLOR = (70, 130, 180)
    BUTTON_HOVER_COLOR = (100, 160, 210)
    HIGHLIGHT_COLOR = (255, 215, 0)
    
    def __init__(self, num_players: int = 2):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Poker Training Engine")
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        self.engine = PokerEngine(num_players=num_players)
        self.state = None
        self.running = True
        self.clock = pygame.time.Clock()
        
        self.buttons = []
        self.selected_action = None
        
    def reset_game(self):
        self.state = self.engine.reset()
        self.selected_action = None
        self.create_action_buttons()
    
    def create_action_buttons(self):
        self.buttons = []
        
        if not self.state or not self.state['legal_actions']:
            return
        
        action_names = {
            0: "Fold",
            1: "Check/Call",
            2: "Raise (Small)",
            3: "Raise (Medium)",
            4: "Raise (Large)",
            5: "All-In"
        }
        
        button_width = 150
        button_height = 50
        button_spacing = 20
        start_x = (self.WINDOW_WIDTH - (button_width * len(self.state['legal_actions']) + 
                   button_spacing * (len(self.state['legal_actions']) - 1))) // 2
        start_y = self.WINDOW_HEIGHT - 100
        
        for idx, action in enumerate(self.state['legal_actions']):
            x = start_x + idx * (button_width + button_spacing)
            y = start_y
            
            action_name = action_names.get(action, f"Action {action}")
            
            self.buttons.append({
                'rect': pygame.Rect(x, y, button_width, button_height),
                'action': action,
                'name': action_name
            })
    
    def draw_table(self):
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2
        
        pygame.draw.ellipse(self.screen, self.TABLE_COLOR, 
                          (center_x - 300, center_y - 150, 600, 300))
    
    def draw_player_info(self):
        if not self.state:
            return
        
        player_positions = [
            (self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT - 200),
            (self.WINDOW_WIDTH // 2, 100)
        ]
        
        for idx, agent in enumerate(self.engine.agents):
            pos_x, pos_y = player_positions[idx]
            
            is_current = agent == self.state['current_player']
            color = self.HIGHLIGHT_COLOR if is_current else self.TEXT_COLOR
            
            player_text = self.font_medium.render(f"Player {idx}", True, color)
            text_rect = player_text.get_rect(center=(pos_x, pos_y))
            self.screen.blit(player_text, text_rect)
            
            if is_current:
                pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR, 
                                 (pos_x - 100, pos_y), 10)
    
    def draw_community_cards(self):
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2
        
        card_width = 60
        card_height = 90
        card_spacing = 10
        
        num_cards = 5
        start_x = center_x - (num_cards * (card_width + card_spacing)) // 2
        
        for i in range(num_cards):
            x = start_x + i * (card_width + card_spacing)
            y = center_y - card_height // 2
            
            pygame.draw.rect(self.screen, self.CARD_COLOR, 
                           (x, y, card_width, card_height), border_radius=5)
            pygame.draw.rect(self.screen, (0, 0, 0), 
                           (x, y, card_width, card_height), 2, border_radius=5)
    
    def draw_pot(self):
        if not self.state:
            return
        
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2 + 100
        
        pot_text = self.font_medium.render("Pot", True, self.TEXT_COLOR)
        text_rect = pot_text.get_rect(center=(center_x, center_y))
        self.screen.blit(pot_text, text_rect)
    
    def draw_buttons(self):
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            is_hover = button['rect'].collidepoint(mouse_pos)
            color = self.BUTTON_HOVER_COLOR if is_hover else self.BUTTON_COLOR
            
            pygame.draw.rect(self.screen, color, button['rect'], border_radius=10)
            pygame.draw.rect(self.screen, self.TEXT_COLOR, button['rect'], 2, border_radius=10)
            
            text = self.font_small.render(button['name'], True, self.TEXT_COLOR)
            text_rect = text.get_rect(center=button['rect'].center)
            self.screen.blit(text, text_rect)
    
    def draw_game_status(self):
        if not self.state:
            return
        
        if self.engine.is_done():
            status_text = "Game Over - Press R to restart"
            color = self.HIGHLIGHT_COLOR
        else:
            status_text = f"Current: {self.state['current_player']}"
            color = self.TEXT_COLOR
        
        text = self.font_small.render(status_text, True, color)
        text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, 30))
        self.screen.blit(text, text_rect)
    
    def handle_button_click(self, pos):
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                self.selected_action = button['action']
                return True
        return False
    
    def process_action(self):
        if self.selected_action is not None and not self.engine.is_done():
            self.state = self.engine.step(self.selected_action)
            self.selected_action = None
            self.create_action_buttons()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.reset_game()
                elif event.key == pygame.K_q:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    if self.state and self.state['legal_actions']:
                        self.selected_action = np.random.choice(self.state['legal_actions'])
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_button_click(event.pos)
    
    def draw(self):
        self.screen.fill(self.BACKGROUND_COLOR)
        
        self.draw_table()
        self.draw_community_cards()
        self.draw_pot()
        self.draw_player_info()
        self.draw_buttons()
        self.draw_game_status()
        
        help_text = self.font_small.render("R: Reset | SPACE: Random Action | Q: Quit", 
                                          True, self.TEXT_COLOR)
        self.screen.blit(help_text, (10, self.WINDOW_HEIGHT - 30))
        
        pygame.display.flip()
    
    def run(self):
        self.reset_game()
        
        while self.running:
            self.handle_events()
            self.process_action()
            self.draw()
            self.clock.tick(60)
        
        self.engine.close()
        pygame.quit()
