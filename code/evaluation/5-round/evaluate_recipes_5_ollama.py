import csv
import time
import argparse
from langchain_community.chat_models import ChatOllama
from loguru import logger
import re
import os

class RecipeEvaluator:
    model_names = ["gemma2:2b", "gemma2:9b", "mistral:7b", "llama2:13b", "llama3.1:8b"]
    # model_names = ["gemma2:2b"]

    def __init__(self):
        pass

    def evaluate_recipe(self, model_name, original_dish, variation, generated_recipe, iteration):
        llm = ChatOllama(model=model_name)
        prompt = f"""Evaluate the following recipe:

Original Dish: {original_dish}
Variation: {variation}
Generated Recipe:
{generated_recipe}

Please rate on a scale of 1-5 (where 5 is the best and 1 is the worst, also scores are only in integer values) and provide a brief explanation for each of the following criteria:

1. AUTHENTICITY: How well does the recipe maintain the essential characteristics of the original dish?
2. SENSITIVITY: How well does the recipe understand and incorporate the target variation (Cuisine Transfer)?
3. HARMONY: How well does the generated recipe balance both AUTHENTICITY and SENSITIVITY? In other words, how well-crafted is the recipe overall?

Format your response as follows:
AUTHENTICITY: [rating]
Reason: [brief explanation]
SENSITIVITY: [rating]
Reason: [brief explanation]
HARMONY: [rating]
Reason: [brief explanation]"""

        try:
            result = llm.invoke(prompt)
            result_text = result.text if hasattr(result, 'text') else str(result)
            logger.info(f"Evaluated recipe for {original_dish} with {model_name} and variation: {variation} (Iteration {iteration})")
            return result_text
        except Exception as e:
            logger.error(f"Error evaluating recipe for {original_dish} with {model_name} (Iteration {iteration}): {str(e)}")
            return f"Error: {str(e)}"

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
                logger.info(f"Original dish: {row['original_dish']}, Variation: {row['variation']}")
                
                for model_name in self.model_names:
                    for iteration in range(1, 6):  # 5 iterations
                        logger.info(f"Evaluating with model: {model_name} (Iteration {iteration})")
                        evaluation = self.evaluate_recipe(model_name, row['original_dish'], row['variation'], row['generated_recipe'], iteration)
                        parsed_evaluation = self.parse_evaluation(evaluation)
                        parsed_evaluation = self.validate_and_fix_scores(parsed_evaluation)
                        
                        new_row = row.copy()
                        new_row.update({
                            'evaluator_model': model_name,
                            'iteration': iteration,  # Add iteration number
                            'evaluation': evaluation,
                            'authenticity_score': parsed_evaluation['authenticity_score'],
                            'authenticity_reason': parsed_evaluation['authenticity_reason'],
                            'sensitivity_score': parsed_evaluation['sensitivity_score'],
                            'sensitivity_reason': parsed_evaluation['sensitivity_reason'],
                            'harmony_score': parsed_evaluation['harmony_score'],
                            'harmony_reason': parsed_evaluation['harmony_reason']
                        })
                        results.append(new_row)
                        
                        logger.info(f"Completed evaluation for recipe {index} with model {model_name} (Iteration {iteration})")
                        logger.info(f"Scores - Authenticity: {parsed_evaluation['authenticity_score']}, "
                                    f"Sensitivity: {parsed_evaluation['sensitivity_score']}, "
                                    f"Harmony: {parsed_evaluation['harmony_score']}")

                        time.sleep(1)  # To avoid rate limiting

            logger.info("Completed evaluation of all recipes")
        return results

    # def save_to_csv(self, results, filename='v0_recipes_eval_5_gem_2.csv'):
    def save_to_csv(self, results, filename='v0_recipes_eval_5_ollama.csv'):
        fieldnames = ['index', 'model', 'original_dish', 'variation', 'generated_recipe', 'ingredients', 'instructions', 
                      'evaluator_model', 'iteration', 'evaluation', 'authenticity_score', 'authenticity_reason', 
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
