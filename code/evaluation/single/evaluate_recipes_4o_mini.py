# When running the script use: python3 evaluate_recipes.py (input file)
# i.e. "python3 evaluate_recipes.py generated_recipes.csv"

import csv
import time
import argparse
from loguru import logger
import re
import openai
import os

class RecipeEvaluator:
    def __init__(self):
        api_key_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../API_KEY/API_KEY_openai.txt")
        try:
            with open(api_key_path, "r") as f:
                openai.api_key = f.read().strip()
            if not openai.api_key:
                raise ValueError("API key is empty")
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
        except Exception as e:
            raise Exception(f"Error reading API key: {str(e)}")

    def evaluate_recipe(self, original_dish, variation, generated_recipe):
        prompt = f"""Evaluate the following recipe:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Please rate on a scale of 1-5 (where 5 is the best and 1 is the worst, also scores are only in integer values) and provide a brief explanation for each of the following criteria:

1. AUTHENTICITY: How well does the recipe maintain the essential characteristics of the original dish?
   Example: For a request to create a Korean-style spaghetti recipe, an extremely poor case would be generating a recipe for japchae.

2. SENSITIVITY: How well does the recipe understand and incorporate the target variation (Cuisine Transfer)?
   Example: For a request to create a halal version of spaghetti, an extremely incorrect case would be including pork as an ingredient.

3. HARMONY: How well does the generated recipe balance both AUTHENTICITY and SENSITIVITY? In other words, how well-crafted is the recipe overall?

Format your response as follows:
AUTHENTICITY: [rating]
Reason: [brief explanation]
SENSITIVITY: [rating]
Reason: [brief explanation]
HARMONY: [rating]
Reason: [brief explanation]"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"Error in evaluation: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in GPT-4o mini evaluation: {str(e)}")
            return f"Unexpected error in evaluation: {str(e)}"

    def parse_evaluation(self, evaluation):
        patterns = [
            (r'AUTHENTICITY:?\s*(?:\*|\#)?\s*(\d+(?:\.\d+)?)', 'authenticity_score'),
            (r'AUTHENTICITY:.*?(?:Reason|Explanation):\s*(.*?)(?=\n*(?:SENSITIVITY|HARMONY|$))', 'authenticity_reason'),
            (r'SENSITIVITY:?\s*(?:\*|\#)?\s*(\d+(?:\.\d+)?)', 'sensitivity_score'),
            (r'SENSITIVITY:.*?(?:Reason|Explanation):\s*(.*?)(?=\n*(?:HARMONY|$))', 'sensitivity_reason'),
            (r'HARMONY:?\s*(?:\*|\#)?\s*(\d+(?:\.\d+)?)', 'harmony_score'),
            (r'HARMONY:.*?(?:Reason|Explanation):\s*(.*?)(?=$)', 'harmony_reason')
        ]

        result = {}
        for pattern, key in patterns:
            match = re.search(pattern, evaluation, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Remove any markdown characters and convert to float for scores
                if 'score' in key:
                    value = re.sub(r'[^\d.]', '', value)
                    try:
                        value = float(value)
                    except ValueError:
                        value = None
                result[key] = value
            else:
                result[key] = None

        return result

    def validate_and_fix_scores(self, parsed_evaluation):
        for key in ['authenticity_score', 'sensitivity_score', 'harmony_score']:
            score = parsed_evaluation[key]
            if score is None or score < 1 or score > 5:
                text = parsed_evaluation[key.replace('score', 'reason')]
                if text:
                    match = re.search(r'\d+(?:\.\d+)?', text)
                    if match:
                        try:
                            score = float(match.group())
                            if 1 <= score <= 5:
                                parsed_evaluation[key] = score
                            else:
                                parsed_evaluation[key] = None
                        except ValueError:
                            parsed_evaluation[key] = None
                    else:
                        parsed_evaluation[key] = None
                else:
                    parsed_evaluation[key] = None
        return parsed_evaluation

    def evaluate_recipes(self, input_filename):
        results = []
        with open(input_filename, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            total_rows = sum(1 for row in reader)
            file.seek(0)
            next(reader)
            
            logger.info(f"Starting evaluation of {total_rows} recipes")
            for index, row in enumerate(reader, start=1):
                logger.info(f"Evaluating recipe {index}/{total_rows}")
                logger.info(f"Model: {row['model']}, Original dish: {row['original_dish']}, Variation: {row['variation']}")
                
                evaluation = self.evaluate_recipe(row['original_dish'], row['variation'], row['generated_recipe'])
                parsed_evaluation = self.parse_evaluation(evaluation)
                parsed_evaluation = self.validate_and_fix_scores(parsed_evaluation)
                
                row.update({
                    'evaluation': evaluation,
                    'authenticity_score': parsed_evaluation['authenticity_score'],
                    'authenticity_reason': parsed_evaluation['authenticity_reason'],
                    'sensitivity_score': parsed_evaluation['sensitivity_score'],
                    'sensitivity_reason': parsed_evaluation['sensitivity_reason'],
                    'harmony_score': parsed_evaluation['harmony_score'],
                    'harmony_reason': parsed_evaluation['harmony_reason']
                })
                results.append(row)
                
                logger.info(f"Completed evaluation for recipe {index}")
                logger.info(f"Scores - Authenticity: {parsed_evaluation['authenticity_score']}, "
                            f"Sensitivity: {parsed_evaluation['sensitivity_score']}, "
                            f"Harmony: {parsed_evaluation['harmony_score']}")
                
                time.sleep(1)  # To avoid rate limiting
            
            logger.info("Completed evaluation of all recipes")
        return results
    
    def save_to_csv(self, results, filename='v0_recipes_eval_4o_mini.csv'):
        fieldnames = ['index', 'model', 'original_dish', 'variation', 'generated_recipe', 'ingredients', 'instructions', 
                    'evaluation', 'authenticity_score', 'authenticity_reason', 'sensitivity_score', 
                    'sensitivity_reason', 'harmony_score', 'harmony_reason']
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated recipes")
    parser.add_argument("input_file", help="Input CSV file containing generated recipes")
    args = parser.parse_args()

    logger.info(f"Starting recipe evaluation process for file: {args.input_file}")
    start_time = time.time()

    evaluator = RecipeEvaluator()
    results = evaluator.evaluate_recipes(args.input_file)
    evaluator.save_to_csv(results)

    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"Recipe evaluation completed. Total execution time: {total_time:.2f} seconds")
    print(f"Recipe evaluation completed. Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()