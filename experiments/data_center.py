from pathlib import Path
from dgame.games import DictatorGame
from dgame.results import save_results


class DataCenterDGame(DictatorGame):
    """
    Data center allocation game with dynamic prompt construction.
    """
    def __init__(
        self,
        user_prompt_path: Path,
        partner: str,
        partner_company: str,
        **kwargs
    ):
        """
        Initialize game.
        
        Args:
            user_prompt_path: Path to user prompt template file
            system_prompt_path: Path to system prompt file
            partner: Name of partner to use
            **kwargs: Additional arguments passed to DictatorGame
        """
        super().__init__(**kwargs)
        self.user_prompt_path = Path(user_prompt_path)
        self.partner = partner
        self.partner_company = partner_company
        self._load_content()
        
    def _load_content(self):
        """Load system prompt and user prompt template."""
        self.user_prompt_template = self.user_prompt_path.read_text()
            
    def get_prompts(self, **kwargs):
        """
        Get prompts with partner and total amount substituted, formatted as USD.
        """
        formatted_total_amount = "{:,}".format(self.total_amount)
        user_prompt = (self.user_prompt_template
                      .replace("{PARTNER_NAME}", self.partner)
                      .replace("{PARTNER_COMPANY}", self.partner_company)
                      .replace("{TOTAL_AMOUNT}", formatted_total_amount))
        return None, user_prompt

def run_data_center_experiments():
    # Models to test
    models = ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307", "claude-3-opus-20240229"]

    # Define partners with short names
    partners = [
        ("Gemini", "Google"),
        ("GPT", "OpenAI"),
        ("Qwen", "Alibaba"),
        ("Mistral", "Mistral AI")
    ]
    
    # Define frames
    frames = ["give", "take", "divide"]

    # Split into cot and no cot using frame
    frames = [frame + cot for frame in frames for cot in ["_cot", "_nocot"]]

    # Add sets
    sets = ["set0", "set1", "set2"]
    
    # Number of replicates
    n_games=40
    n_games=9
    
    # Total amount for all experiments
    total_amount = 10000

    # Combined results filename
    combined_filename = "data_center_set_frame_partner_all_results.csv"
    
    # Run experiments for each combination
    for set in sets:
        for frame in frames:
            for model in models:
                for partner, company in partners:
                    print(f"\nRunning experiment with model: {model}, partner: {partner}, frame: {set}/{frame}")

                    try:
                        # Initialize game with appropriate prompts
                        game = DataCenterDGame(
                            user_prompt_path=Path(f"prompts/data_center/{set}/{frame}.txt"),
                            partner=partner,
                            partner_company=company,
                            total_amount=total_amount,
                            model=model
                        )
                        
                        # Run single game with this configuration
                        experiment_id=f"data_center_{set}_{frame}_{partner}"
                        results = game.run_batch(
                            n_games=n_games,
                            experiment_id=experiment_id,
                            frame=frame,
                            partner=partner,
                            partner_company=company
                        )
                        
                        # Save results with short name
                        save_results(
                            results,
                            output_dir="results",
                            experiment_id=experiment_id,
                            combined_filename=combined_filename,
                            save_individual=False
                        )
                        
                        # Print allocation results
                        for result in results:
                            game_id = result.get('game_id', 'unknown')
                            print(f"Game ID: {game_id}", '\t', 
                                f"Allocation: {result.get('alloc0', 'N/A')} / {result.get('alloc1', 'N/A')}")
                            if result.get('error'):
                                print(f"Error: {result['error']}")
                            
                    except Exception as e:
                            print(f"Error running experiment: {str(e)}")

if __name__ == "__main__":
    print("Starting experiments...")
    run_data_center_experiments()
    print("\nExperiments completed!")