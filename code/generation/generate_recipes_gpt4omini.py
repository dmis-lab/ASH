import openai
import os
import csv
import time
import re
import argparse
from loguru import logger

class RecipeGenerator:
    model_name = "gpt-4o-mini"
    dishes = [
        "Fried Rice", "Sandwich", "Soup Noodle", "Savoury Pie", "Fried Noodles",
        "Rolls", "Savory Waffle", "Fried Chicken", "Barbecued Meat", "French Fries",
        "Burger", "Pasta", "Pancake", "Stew", "Pizza", "Burritos", "Crepes",
        "Lasagna", "Curry", "Salad"
    ]

    variations = [
        # Regional (30)
        'Japanese', 'Korean', 'Chinese', 'Thai', 'Vietnamese', 'Filipino', 'Indian', 'Russian',
        'Italian', 'French', 'Spanish', 'British', 'Irish', 'Greek', 'Scottish', 'Swedish',
        'Southern US', 'Brazilian', 'Mexican', 'Jamaican', 'Hawaiian', 'Costa Rican', 'Canadian', 'Peruvian',
        'Moroccan', 'Ethiopian', 'Algerian', 'Egyptian', 'Australian', 'Polynesian',
        # Religious (6)
        'Buddhist', 'Hindu diet', 'Islamic diet', 'Jain diet', 'Kosher', 'Zoroastrian',
        # Historical (4)
        'Aztec', 'Medieval', 'Byzantine', 'Ottoman'
    ]

    def __init__(self):
        self.index = 1
        api_key_path = os.path.join(os.path.dirname(__file__), '../API_KEY', 'API_KEY_openai.txt')
        try:
            with open(api_key_path, 'r') as f:
                openai.api_key = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading API key from {api_key_path}: {str(e)}")
            raise e

    def generate_recipe(self, dish, variation):
        prompt = f"""Can you apply the elements of {variation} cuisine to this dish and make it into a recipe?
Dish: {dish}
The response should be in the following form for ingredients and instructions each. For example:
ingredients: 
<<ingredient1>>,
<<ingredient2>>,
...

instructions: 
1. <<instruction1>>
2. <<instruction2>>
...
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates recipes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            result_text = response['choices'][0]['message']['content']
            logger.info(f"Generated recipe for '{dish}' with variation: '{variation}'")
            return result_text
        except Exception as e:
            logger.error(f"Error generating recipe for '{dish}' with variation '{variation}': {str(e)}")
            return f"Error: {str(e)}"

    def extract_ingredients_instructions(self, result):
        ingredients_match = re.search(r'ingredients:\s*{(.*?)}', result, re.IGNORECASE | re.DOTALL)
        instructions_match = re.search(r'instructions:\s*{(.*?)}', result, re.IGNORECASE | re.DOTALL)

        ingredients = ingredients_match.group(1).strip() if ingredients_match else ""
        instructions = instructions_match.group(1).strip() if instructions_match else ""

        ingredients_list = [line.strip().lstrip('0123456789. *-') for line in ingredients.split('\n') if line.strip()]
        instructions_list = [line.strip().lstrip('0123456789. *-') for line in instructions.split('\n') if line.strip()]

        ingredients = ', '.join(ingredients_list)
        instructions = '\n'.join(instructions_list)

        return ingredients, instructions

    def generate_recipes(self):
        results = []
        for dish in self.dishes:
            for variation in self.variations:
                generated_recipe = self.generate_recipe(dish, variation)
                ingredients, instructions = self.extract_ingredients_instructions(generated_recipe)

                results.append({
                    'index': self.index,
                    'model': self.model_name,
                    'original_dish': dish,
                    'variation': variation,
                    'generated_recipe': generated_recipe,
                    'ingredients': ingredients,
                    'instructions': instructions
                })
                self.index += 1
                time.sleep(1)  # To avoid rate limiting
        return results

    def save_to_csv(self, results, filename='generated_recipes_gpt4omini.csv'):
        fieldnames = ['index', 'model', 'original_dish', 'variation', 'generated_recipe', 'ingredients', 'instructions']
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate recipes with cultural, religious, and historical variations using OpenAI API")
    args = parser.parse_args()

    start_time = time.time()

    generator = RecipeGenerator()
    results = generator.generate_recipes()
    generator.save_to_csv(results)

    end_time = time.time()
    total_time = end_time - start_time

    logger.info(f"Total execution time: {total_time:.2f} seconds")
    print(f"Recipe generation completed. Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
