import google.generativeai as genai
import os
import csv
import time
import argparse
from loguru import logger
import re

class RecipeEvaluator:
    def __init__(self):
        api_key_path = '../API_KEY/API_KEY_gemini.txt'
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key is empty")
            genai.configure(api_key=api_key)
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
        except Exception as e:
            raise Exception(f"Error reading API key: {str(e)}")

    def evaluate_recipe(self, original_dish, variation, generated_recipe, iteration):
        prompt = f"""Evaluate the following recipe:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Please rate on a scale of 1-5 (where 5 is the best and 1 is the worst, also scores are only in integer values) and provide a brief explanation for each of the following criteria:

1. AUTHENTICITY: How well does the recipe maintain the essential characteristics of the original dish?
2. SENSITIVITY: How well does the recipe incorporate the target variation (Cuisine Transfer)?
3. HARMONY: How well does the recipe balance both AUTHENTICITY and SENSITIVITY?

Format your response as follows:
AUTHENTICITY: [rating]
Reason: [brief explanation]
SENSITIVITY: [rating]
Reason: [brief explanation]
HARMONY: [rating]
Reason: [brief explanation]"""

        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            logger.info(f"Evaluated recipe for {original_dish} with variation {variation} (Iteration {iteration})")
            return response.text
        except Exception as e:
            logger.error(f"Unexpected Gemini API error: {str(e)}")
            return f"Error in evaluation: {str(e)}"

    def parse_evaluation(self, evaluation):
        patterns = [
            (r'AUTHENTICITY:?\s*(\d+)', 'authenticity_score'),
            (r'AUTHENTICITY.*?(?:Reason):\s*(.*?)(?=\n*SENSITIVITY)', 'authenticity_reason'),
            (r'SENSITIVITY:?\s*(\d+)', 'sensitivity_score'),
            (r'SENSITIVITY.*?(?:Reason):\s*(.*?)(?=\n*HARMONY)', 'sensitivity_reason'),
            (r'HARMONY:?\s*(\d+)', 'harmony_score'),
            (r'HARMONY.*?(?:Reason):\s*(.*)', 'harmony_reason')
        ]

        result = {}
        for pattern, key in patterns:
            match = re.search(pattern, evaluation, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
            else:
                result[key] = None

        return result

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
  
              for iteration in range(1, 6):  # Repeat evaluation 5 times
                  logger.info(f"Iteration {iteration}")
  
                  row_copy = row.copy()
  
                  evaluation = self.evaluate_recipe(row['original_dish'], row['variation'], row['generated_recipe'], iteration)
                  parsed_evaluation = self.parse_evaluation(evaluation)
  
                  row_copy.update({
                      'iteration': iteration,  # Save iteration number
                      'evaluator_model': 'gemini-1.5-flash',  # Change evaluator_model for different models!
                      'evaluation': evaluation,
                      'authenticity_score': parsed_evaluation['authenticity_score'],
                      'authenticity_reason': parsed_evaluation['authenticity_reason'],
                      'sensitivity_score': parsed_evaluation['sensitivity_score'],
                      'sensitivity_reason': parsed_evaluation['sensitivity_reason'],
                      'harmony_score': parsed_evaluation['harmony_score'],
                      'harmony_reason': parsed_evaluation['harmony_reason']
                  })
                  results.append(row_copy)
  
                  logger.info(f"Completed evaluation for recipe {index} (Iteration {iteration})")
                  time.sleep(1)  # To avoid rate limiting
  
          logger.info(f"Completed evaluation of all {total_rows} recipes")
      return results


    def save_to_csv(self, results, filename='v0_recipes_eval_5_gem_15_flash.csv'):
        fieldnames = ['index', 'model', 'evaluator_model', 'iteration', 'original_dish', 'variation', 'generated_recipe', 
                      'ingredients', 'instructions', 'evaluation', 'authenticity_score', 'authenticity_reason', 
                      'sensitivity_score', 'sensitivity_reason', 'harmony_score', 'harmony_reason']
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
