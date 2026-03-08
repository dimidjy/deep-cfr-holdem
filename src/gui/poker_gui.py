import pygame
import numpy as np
from typing import Dict, List, Optional, Any
from src.engine import PokerEngine


class CardRenderer:
    
    @staticmethod
    def parse_card(card_str: str) -> tuple:
        if not card_str or len(card_str) < 2:
            return ('?', '?', (0, 0, 0))
        
        rank = card_str[:-1]
        suit = card_str[-1]
        
        if rank == 'T':
            rank = '10'
        
        color = (220, 20, 20) if suit in ['H', 'D'] else (0, 0, 0)
        
        return (rank, suit, color)


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
    
    def __init__(self, num_players: int = 2, ai_model=None):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        if ai_model:
            pygame.display.set_caption("Poker vs Trained AI")
        else:
            pygame.display.set_caption("Poker Training Engine")
        
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_card = pygame.font.Font(None, 32)
        
        self.engine = PokerEngine(num_players=num_players)
        self.state = None
        self.running = True
        self.clock = pygame.time.Clock()
        
        self.buttons = []
        self.selected_action = None
        self.card_renderer = CardRenderer()
        self.human_player = 'player_0'
        self.action_history = []
        self.current_round = "Pre-Flop"
        self.last_action_text = ""
        self.hand_count = 0
        self.wins = 0
        self.losses = 0
        self.result_counted = False
        
        self.ai_model = ai_model
        self.ai_name = "Trained AI" if ai_model else "AI (Random)"
        self.show_opponent_cards = False
        
    def reset_game(self):
        self.state = self.engine.reset()
        self.selected_action = None
        self.action_history = []
        self.current_round = "Pre-Flop"
        self.last_action_text = "New hand started"
        self.hand_count += 1
        self.result_counted = False
        self.create_action_buttons()
        self.add_action_to_history(f"--- Hand #{self.hand_count} ---", "")
    
    def create_action_buttons(self):
        self.buttons = []
        
        if not self.state or not self.state['legal_actions']:
            return
        
        action_names = {
            0: "Fold",
            1: "Check/Call",
            2: "Raise Half Pot",
            3: "Raise Full Pot",
            4: "All-In"
        }
        
        button_width = 150
        button_height = 50
        button_spacing = 20
        start_x = (self.WINDOW_WIDTH - (button_width * len(self.state['legal_actions']) + 
                   button_spacing * (len(self.state['legal_actions']) - 1))) // 2
        start_y = self.WINDOW_HEIGHT - 80
        
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
    
    def _player_positions(self):
        """Return (x, y) seat positions for up to 6 players, index 0 = human (bottom center)."""
        n = len(self.engine.agents)
        cx, cy = self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2
        # Positions for each seat count
        positions_map = {
            2: [(cx, self.WINDOW_HEIGHT - 200), (cx, 120)],
            3: [(cx, self.WINDOW_HEIGHT - 200), (cx + 320, 200), (cx - 320, 200)],
            4: [(cx, self.WINDOW_HEIGHT - 200), (cx + 380, cy), (cx, 120), (cx - 380, cy)],
            5: [(cx, self.WINDOW_HEIGHT - 200), (cx + 350, self.WINDOW_HEIGHT - 350),
                (cx + 350, 200), (cx - 350, 200), (cx - 350, self.WINDOW_HEIGHT - 350)],
            6: [(cx, self.WINDOW_HEIGHT - 200), (cx + 380, self.WINDOW_HEIGHT - 350),
                (cx + 380, 200), (cx, 120), (cx - 380, 200), (cx - 380, self.WINDOW_HEIGHT - 350)],
        }
        return positions_map.get(n, positions_map[6])

    def draw_player_info(self):
        if not self.state:
            return

        positions = self._player_positions()

        for idx, agent in enumerate(self.engine.agents):
            if idx >= len(positions):
                break
            pos_x, pos_y = positions[idx]

            is_current = agent == self.state['current_player']
            color = self.HIGHLIGHT_COLOR if is_current else self.TEXT_COLOR

            player_name = "You" if agent == self.human_player else f"AI-{idx}"
            chips = self.engine.get_player_chips(agent)

            player_text = self.font_medium.render(player_name, True, color)
            text_rect = player_text.get_rect(center=(pos_x, pos_y))
            self.screen.blit(player_text, text_rect)

            chips_text = self.font_small.render(f"Chips: {chips}", True, self.TEXT_COLOR)
            chips_rect = chips_text.get_rect(center=(pos_x, pos_y + 25))
            self.screen.blit(chips_text, chips_rect)

            if is_current:
                pygame.draw.circle(self.screen, self.HIGHLIGHT_COLOR,
                                   (pos_x - 80, pos_y), 10)

            player_cards = self.engine.get_player_cards(agent)
            if player_cards:
                card_width = 60
                card_height = 85
                card_spacing = 8
                start_x = pos_x - (len(player_cards) * (card_width + card_spacing)) // 2

                y_offset = 50 if agent == self.human_player else 45

                for card_idx, card_str in enumerate(player_cards):
                    x = start_x + card_idx * (card_width + card_spacing)
                    y = pos_y + y_offset

                    if agent == self.human_player or self.show_opponent_cards:
                        self.draw_card(x, y, card_str, card_width, card_height)
                    else:
                        self.draw_card(x, y, "", card_width, card_height, face_down=True)
    
    def draw_card(self, x: int, y: int, card_str: str = "", width: int = 60, height: int = 90, face_down: bool = False):
        pygame.draw.rect(self.screen, self.CARD_COLOR, 
                       (x, y, width, height), border_radius=5)
        pygame.draw.rect(self.screen, (0, 0, 0), 
                       (x, y, width, height), 2, border_radius=5)
        
        if face_down:
            back_color = (70, 130, 180)
            pygame.draw.rect(self.screen, back_color, 
                           (x + 5, y + 5, width - 10, height - 10), border_radius=3)
        elif card_str:
            rank, suit_letter, color = self.card_renderer.parse_card(card_str)
            
            rank_text = self.font_large.render(rank, True, color)
            suit_text = self.font_large.render(suit_letter, True, color)
            
            rank_rect = rank_text.get_rect(topleft=(x + 8, y + 8))
            suit_rect = suit_text.get_rect(center=(x + width // 2, y + height - 35))
            
            self.screen.blit(rank_text, rank_rect)
            self.screen.blit(suit_text, suit_rect)
    
    def draw_community_cards(self):
        if not self.state:
            return
        
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2
        
        card_width = 60
        card_height = 90
        card_spacing = 10
        
        community_cards = self.engine.get_community_cards()
        
        num_cards = 5
        start_x = center_x - (num_cards * (card_width + card_spacing)) // 2
        
        for i in range(num_cards):
            x = start_x + i * (card_width + card_spacing)
            y = center_y - card_height // 2
            
            if i < len(community_cards):
                self.draw_card(x, y, community_cards[i], card_width, card_height)
            else:
                self.draw_card(x, y, "", card_width, card_height)
    
    def draw_pot(self):
        if not self.state:
            return
        
        center_x = self.WINDOW_WIDTH // 2
        center_y = self.WINDOW_HEIGHT // 2 + 100
        
        pot_size = self.engine.get_pot_size()
        
        pot_text = self.font_medium.render(f"Pot: {pot_size}", True, self.TEXT_COLOR)
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
    
    def add_action_to_history(self, player: str, action_text: str):
        timestamp = len(self.action_history)
        self.action_history.append(f"{player}: {action_text}")
        if len(self.action_history) > 8:
            self.action_history.pop(0)
    
    def update_round_info(self):
        self.current_round = self.engine.get_stage()
    
    def draw_game_status(self):
        if not self.state:
            return
        
        if self.engine.is_done():
            rewards = self.engine.get_rewards()
            human_reward = rewards.get(self.human_player, 0)
            
            if not self.result_counted:
                if human_reward > 0:
                    self.wins += 1
                elif human_reward < 0:
                    self.losses += 1
                self.result_counted = True
            
            if human_reward > 0:
                status_text = f"YOU WON! +{int(human_reward)} chips"
                detail_text = f"Press R for next hand | Record: {self.wins}W - {self.losses}L"
            elif human_reward < 0:
                status_text = f"YOU LOST {int(human_reward)} chips"
                detail_text = f"Press R for next hand | Record: {self.wins}W - {self.losses}L"
            else:
                status_text = "TIE!"
                detail_text = f"Press R for next hand | Record: {self.wins}W - {self.losses}L"
            
            color = self.HIGHLIGHT_COLOR
            
            text = self.font_large.render(status_text, True, color)
            text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, 30))
            self.screen.blit(text, text_rect)
            
            detail = self.font_small.render(detail_text, True, self.TEXT_COLOR)
            detail_rect = detail.get_rect(center=(self.WINDOW_WIDTH // 2, 65))
            self.screen.blit(detail, detail_rect)
        else:
            if self.state['current_player'] == self.human_player:
                status_text = "YOUR TURN"
                color = self.HIGHLIGHT_COLOR
                detail_text = "Choose your action below"
            else:
                status_text = "AI THINKING..."
                color = (200, 200, 200)
                detail_text = "Waiting for opponent"
            
            text = self.font_medium.render(status_text, True, color)
            text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, 25))
            self.screen.blit(text, text_rect)
            
            detail = self.font_small.render(detail_text, True, self.TEXT_COLOR)
            detail_rect = detail.get_rect(center=(self.WINDOW_WIDTH // 2, 55))
            self.screen.blit(detail, detail_rect)
            
            round_text = self.font_small.render(f"Round: {self.current_round}", True, self.HIGHLIGHT_COLOR)
            round_rect = round_text.get_rect(topright=(self.WINDOW_WIDTH - 20, 20))
            self.screen.blit(round_text, round_rect)
    
    def draw_action_history(self):
        history_x = 20
        history_y = 80
        
        title = self.font_small.render("Action History:", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (history_x, history_y))
        
        bg_rect = pygame.Rect(history_x - 5, history_y + 25, 280, 200)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
        pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, bg_rect, 2, border_radius=5)
        
        font_tiny = pygame.font.Font(None, 20)
        
        for idx, action_text in enumerate(self.action_history[-8:]):
            y_pos = history_y + 30 + (idx * 22)
            
            if action_text.startswith("---"):
                color = self.HIGHLIGHT_COLOR
            elif action_text.startswith("You:"):
                color = (100, 255, 100)
            elif action_text.startswith("AI:"):
                color = (255, 150, 150)
            else:
                color = (200, 200, 200)
            
            text = font_tiny.render(action_text, True, color)
            self.screen.blit(text, (history_x + 5, y_pos))
    
    def draw_game_info(self):
        info_x = self.WINDOW_WIDTH - 300
        info_y = 80
        
        title = self.font_small.render("Game Info:", True, self.HIGHLIGHT_COLOR)
        self.screen.blit(title, (info_x, info_y))
        
        bg_rect = pygame.Rect(info_x - 5, info_y + 25, 280, 200)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
        pygame.draw.rect(self.screen, self.HIGHLIGHT_COLOR, bg_rect, 2, border_radius=5)
        
        font_info = pygame.font.Font(None, 22)
        
        pot = self.engine.get_pot_size()
        your_chips = self.engine.get_player_chips(self.human_player)

        info_lines = [
            f"Hand: #{self.hand_count}",
            f"Record: {self.wins}W - {self.losses}L",
            f"Pot: {pot} chips",
            f"Your chips: {your_chips}",
        ]
        for agent in self.engine.agents:
            if agent != self.human_player:
                idx = self.engine.agents.index(agent)
                info_lines.append(f"AI-{idx} chips: {self.engine.get_player_chips(agent)}")
        
        for idx, line in enumerate(info_lines):
            y_pos = info_y + 35 + (idx * 22)
            text = font_info.render(line, True, self.TEXT_COLOR)
            self.screen.blit(text, (info_x + 5, y_pos))
    
    def handle_button_click(self, pos):
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                self.selected_action = button['action']
                return True
        return False
    
    def get_action_name(self, action: int) -> str:
        action_names = {
            0: "Fold",
            1: "Check/Call",
            2: "Raise Half Pot",
            3: "Raise Full Pot",
            4: "All-In"
        }
        return action_names.get(action, f"Action {action}")
    
    def process_action(self):
        if self.selected_action is not None and not self.engine.is_done():
            action_name = self.get_action_name(self.selected_action)
            self.add_action_to_history("You", action_name)
            self.last_action_text = f"You {action_name}"

            self.state = self.engine.step(self.selected_action)
            self.selected_action = None
            self.update_round_info()
            self.create_action_buttons()

        # Loop through consecutive AI turns until it's the human's turn (or game over)
        while (not self.engine.is_done()
               and self.state
               and self.state['current_player'] != self.human_player):
            if not self.state['legal_actions']:
                break

            current_agent = self.state['current_player']
            agent_idx = self.engine.agents.index(current_agent)

            if self.ai_model:
                observation = self.engine.get_observation()
                legal_actions = self.state['legal_actions']
                # Use advantage network + regret-matching (current strategy,
                # not the historical average from strategy_network)
                advantages = self.ai_model.compute_counterfactual_values(
                    observation, agent_idx, legal_actions
                )
                pos_adv = np.maximum(advantages, 0.0)
                s = pos_adv.sum()
                if s > 0:
                    strategy = pos_adv / s
                else:
                    mask = np.zeros(self.ai_model.action_size)
                    mask[legal_actions] = 1.0
                    strategy = mask / mask.sum()
                legal_strategy = strategy[legal_actions]
                legal_strategy /= legal_strategy.sum()
                ai_action = np.random.choice(legal_actions, p=legal_strategy)
            else:
                ai_action = np.random.choice(self.state['legal_actions'])

            action_name = self.get_action_name(ai_action)
            pygame.time.delay(300)

            self.add_action_to_history(f"AI-{agent_idx}", action_name)
            self.last_action_text = f"AI-{agent_idx} {action_name}"

            self.state = self.engine.step(ai_action)
            self.update_round_info()
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
                elif event.key == pygame.K_v:
                    self.show_opponent_cards = not self.show_opponent_cards
                    print(f"Show opponent cards: {self.show_opponent_cards}")
            
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
        self.draw_action_history()
        self.draw_game_info()
        
        help_text = self.font_small.render("R: Next Hand | SPACE: Random | V: Show/Hide Cards | Q: Quit", 
                                          True, self.TEXT_COLOR)
        self.screen.blit(help_text, (10, self.WINDOW_HEIGHT - 30))
        
        if self.show_opponent_cards:
            reveal_text = pygame.font.Font(None, 18).render(
                "👁 Opponent cards visible (Press V to hide)", 
                True, (255, 215, 0))
            self.screen.blit(reveal_text, (10, self.WINDOW_HEIGHT - 55))
        
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
