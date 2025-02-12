import os
import csv
import glob
from pathlib import Path

def cleanup_orphaned_responses():
    # Define the base paths
    results_dir = Path('results')
    responses_dir = results_dir / 'responses'
    
    print(f"Looking for CSV files in: {results_dir.absolute()}")
    print(f"Looking for response files in: {responses_dir.absolute()}")
    
    # Get all game IDs from CSV files
    game_ids = set()
    csv_files = list(results_dir.glob('*.csv'))
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"Processing CSV file: {csv_file}")
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Print header to verify game_id column exists
                print(f"CSV headers: {reader.fieldnames}")
                for row in reader:
                    if 'game_id' in row:
                        game_ids.add(row['game_id'])
                    else:
                        print("Warning: 'game_id' column not found in CSV")
                        break
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
    
    print(f"\nCollected {len(game_ids)} unique game IDs")
    
    # Check response files and delete orphans
    response_files = list(responses_dir.glob('*.txt'))
    print(f"\nFound {len(response_files)} response files")
    
    deleted_count = 0
    for response_file in response_files:
        file_id = response_file.stem  # Get filename without extension
        print(f"Checking response file: {file_id}")
        if file_id not in game_ids:
            try:
                response_file.unlink()  # Delete the file
                deleted_count += 1
                print(f"Deleted orphaned response: {response_file.name}")
            except Exception as e:
                print(f"Error deleting {response_file}: {str(e)}")
    
    print(f"\nCleanup complete. Deleted {deleted_count} orphaned response files.")

if __name__ == "__main__":
    cleanup_orphaned_responses()