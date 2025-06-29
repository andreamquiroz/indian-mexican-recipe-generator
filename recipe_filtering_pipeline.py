import pandas as pd
import re
import json
from collections import defaultdict
from typing import List, Dict, Set
import numpy as np


class RecipeCuisineFilter:
    def __init__(self):
        # Mexican cuisine indicators
        self.mexican_ingredients = {
            'primary': [
                'jalapeño', 'jalapeno', 'chipotle', 'poblano', 'serrano', 'habanero',
                'cilantro', 'masa', 'tortilla', 'tomatillo', 'queso fresco',
                'cotija', 'oaxaca cheese', 'mexican crema', 'crema mexicana',
                'achiote', 'epazote', 'mexican oregano', 'chili powder',
                'cumin', 'lime juice', 'avocado', 'black beans', 'pinto beans',
                'chorizo', 'carnitas', 'pozole', 'mole', 'salsa verde',
                'salsa roja', 'pico de gallo', 'guacamole'
            ],
            'secondary': [
                'lime', 'onion', 'garlic', 'tomato', 'bell pepper', 'corn',
                'cheese', 'sour cream', 'hot sauce', 'paprika', 'chili',
                'ground beef', 'chicken breast', 'rice', 'beans'
            ]
        }

        # Indian cuisine indicators
        self.indian_ingredients = {
            'primary': [
                'garam masala', 'turmeric', 'cumin seeds', 'coriander seeds',
                'cardamom', 'cinnamon stick', 'bay leaves', 'curry leaves',
                'mustard seeds', 'fennel seeds', 'fenugreek', 'asafoetida',
                'ghee', 'paneer', 'cottage cheese', 'basmati rice', 'lentils', 'dal', 'chickpeas',
                'coconut milk', 'tamarind', 'jaggery', 'curry powder',
                'tandoori masala', 'biryani', 'naan', 'chapati', 'roti',
                'samosa', 'chutney', 'raita', 'lassi', 'masala chai', 'masala',
                'jeera', 'cumin', 'haldi', 'flour', 'atta', 'besan', 'poha', 'idli',
                'dosa', 'chickpea', 'chana masala', 'dahi',
                'dal', 'rajma', 'bhindi', 'aloo', 'gobi', 'baingan', 'methi',
                'mushroom', 'tandoori', 'pav', 'bhaji', 'gud'
            ],
            'secondary': [
                'yogurt', 'onion', 'garlic', 'ginger', 'tomato', 'oil',
                'salt', 'sugar', 'rice', 'chicken', 'lamb', 'vegetables',
                'cilantro', 'mint', 'lemon juice', 'green chilies', 'nimbu'
            ]
        }

        # Honduran cuisine indicators (smaller set - will need creative approaches)
        self.honduran_ingredients = {
            'primary': [
                'baleada', 'pupusa', 'curtido', 'loroco', 'cuajada', 'mantequilla',
                'plátano verde', 'platano verde', 'yuca', 'casava', 'cassava',
                'frijoles rojos', 'red beans', 'queso seco', 'crema agria',
                'pollo chuco', 'sopa de caracol', 'conch soup', 'machuca',
                'tajadas', 'anafre', 'rosquillas', 'tres leches'
            ],
            'secondary': [
                'plantain', 'banana', 'coconut', 'seafood', 'conch', 'shrimp',
                'fish', 'corn tortilla', 'white cheese', 'sour cream',
                'cilantro', 'onion', 'garlic', 'lime', 'cumin'
            ]
        }

        # Dish name patterns
        self.mexican_dishes = [
            'taco', 'burrito', 'quesadilla', 'enchilada', 'tamale', 'fajita',
            'churro', 'empanada', 'pozole', 'menudo', 'carnitas', 'barbacoa',
            'chilaquiles', 'mole', 'salsa', 'guacamole', 'nachos', 'elote', 'mexican',
            'latin', 'america', 'agujachile', 'flauta', 'cochinita pibil', 'quesadilla',
            'chimichanga', 'chicharrón', 'bolillo', 'chalupa', 'tlacoyo',
            'gringa', 'volcán', 'huitlacoche', 'rajas con crema', 'torta',
            'tostada', 'sopes', 'gordita', 'pico de gallo', 'verde',
            'roja', 'carne', 'pollo', 'asada', 'al pastor', 'lengua', 'cabeza'

        ]

        self.indian_dishes = [
            'curry', 'biryani', 'tandoori', 'masala', 'dal', 'samosa',
            'naan', 'chapati', 'roti', 'dosa', 'idli', 'chutney',
            'raita', 'korma', 'vindaloo', 'tikka', 'kebab', 'pulao',
            'pani', 'bhel', 'puri', 'dhokla', 'upma', 'chole', 'palak',
            'bhindi', 'aloo', 'gobi', 'baingan', 'methi',
            'matar', 'pav', 'bhaji', 'halwa', 'laddu', 'laddoo', 'paratha',
            'bhatura', 'dahi', 'samosa', 'vada', 'kachori', 'indian'
        ]

        self.honduran_dishes = [
            'baleada', 'pupusa', 'machuca', 'sopa de caracol', 'pollo', 'chuco',
            'tajadas', 'casamiento', 'mondongo', 'tres leches', 'latin', 'america',
            'honduran', 'honduras', 'baleadas', 'pupusas', 'curtido',
            'carneada', 'carne asada', 'fried yojoa fish', 'tostones',
            'pescado frito de tajadas', 'tapado olanchano', 'chicharrones',
            'rosquillas', 'plato típico', 'corn tortillas', 'tortilla con quesillo',
            'plantano', 'plantain', 'yuca', 'con', 'chicharrón', 'mojo'
        ]

        self.title_only_mode = True  # when True, ignore ingredients/instructions

    def clean_text(self, text: str) -> str:
        """Clean and normalize text for better matching"""
        if pd.isna(text):
            return ""
        return text.lower().strip()

    def calculate_cuisine_score(self, title: str, ingredients: str, instructions: str, cuisine: str) -> Dict:
        """Calculate cuisine match score based on ingredients and dish names"""
        title_clean = self.clean_text(title)
        ingredients_clean = self.clean_text(ingredients)
        instructions_clean = self.clean_text(instructions)

        # Combine all text for analysis
        if self.title_only_mode:
            ingredients_clean = instructions_clean = ""
            full_text = title_clean
        else:
            full_text = f"{title_clean} {ingredients_clean} {instructions_clean}"

        if cuisine == 'mexican':
            primary_ingredients = self.mexican_ingredients['primary']
            secondary_ingredients = self.mexican_ingredients['secondary']
            dish_names = self.mexican_dishes
        elif cuisine == 'indian':
            primary_ingredients = self.indian_ingredients['primary']
            secondary_ingredients = self.indian_ingredients['secondary']
            dish_names = self.indian_dishes
        elif cuisine == 'honduran':
            primary_ingredients = self.honduran_ingredients['primary']
            secondary_ingredients = self.honduran_ingredients['secondary']
            dish_names = self.honduran_dishes

        # Score calculation
        primary_matches = sum(
            1 for ingredient in primary_ingredients if ingredient in full_text)
        secondary_matches = sum(
            1 for ingredient in secondary_ingredients if ingredient in full_text)
        dish_matches = sum(1 for dish in dish_names if dish in title_clean)

        # Weighted scoring
        total_score = (primary_matches * 3) + \
            (secondary_matches * 1) + (dish_matches * 5)

        if ('spain' in title_clean or 'spanish' in title_clean) and cuisine == 'indian':
            return {
                'primary_matches': 0,
                'secondary_matches': 0,
                'dish_matches': 0,
                'total_score': 0,
                'confidence': 'low'
            }

        return {
            'primary_matches': primary_matches,
            'secondary_matches': secondary_matches,
            'dish_matches': dish_matches,
            'total_score': total_score,
            'confidence': 'high' if primary_matches >= 2 or dish_matches >= 1 else
            'medium' if primary_matches >= 1 and secondary_matches >= 2 else 'low'
        }

    def filter_recipes(self, df: pd.DataFrame, target_per_cuisine: int = 500) -> Dict:
        """Filter recipes for each cuisine"""
        results = {
            'mexican': [],
            'indian': [],
            'honduran': []
        }

        print("Starting recipe filtering...")
        print(f"Total recipes to process: {len(df)}")

        # Process each recipe
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx} recipes...")

            title = row.get('title', '')
            ingredients = row.get('ingredients', '')
            # RecipeNLG uses 'directions'
            instructions = row.get('directions', '')

            # Calculate scores for each cuisine
            mexican_score = self.calculate_cuisine_score(
                title, ingredients, instructions, 'mexican')
            indian_score = self.calculate_cuisine_score(
                title, ingredients, instructions, 'indian')
            honduran_score = self.calculate_cuisine_score(
                title, ingredients, instructions, 'honduran')

            # Determine best match (only if score is above threshold)
            scores = {
                'mexican': mexican_score['total_score'],
                'indian': indian_score['total_score'],
                'honduran': honduran_score['total_score']
            }

            max_cuisine = max(scores, key=scores.get)
            max_score = scores[max_cuisine]

            # Only include if score is above minimum threshold and we need more recipes
            min_score = 3  # Adjust based on results
            if max_score >= min_score and len(results[max_cuisine]) < target_per_cuisine:
                recipe_data = {
                    'title': title,
                    'ingredients': ingredients,
                    'instructions': instructions,
                    'cuisine': max_cuisine,
                    'score': max_score,
                    'confidence': eval(f"{max_cuisine}_score")['confidence'],
                    'original_index': idx
                }
                results[max_cuisine].append(recipe_data)

        return results

    def enhance_honduran_dataset(self, df: pd.DataFrame, results: Dict) -> List:
        """Special handling for Honduran recipes using broader Central American patterns"""
        print("Enhancing Honduran dataset with Central American recipes...")

        # Broader Central American indicators
        central_american_patterns = [
            'plantain', 'banana', 'yuca', 'cassava', 'coconut',
            'central american', 'latin american', 'spanish rice',
            'fried plantain', 'beans and rice', 'seafood soup',
            'tres leches', 'tropical', 'caribbean'
        ]

        additional_honduran = []

        for idx, row in df.iterrows():
            if len(additional_honduran) >= (500 - len(results['honduran'])):
                break

            title = self.clean_text(row.get('title', ''))
            ingredients = self.clean_text(row.get('ingredients', ''))
            instructions = self.clean_text(row.get('directions', ''))

            full_text = f"{title} {ingredients} {instructions}"

            # Check for Central American patterns
            matches = sum(
                1 for pattern in central_american_patterns if pattern in full_text)

            if matches >= 2:  # At least 2 matches
                recipe_data = {
                    'title': row.get('title', ''),
                    'ingredients': row.get('ingredients', ''),
                    'instructions': row.get('directions', ''),
                    'cuisine': 'honduran',
                    'score': matches,
                    'confidence': 'medium',
                    'original_index': idx,
                    'enhanced': True
                }
                additional_honduran.append(recipe_data)

        return additional_honduran


def load_and_filter_recipenlg(file_path: str, target_per_cuisine: int = 500):
    """Main function to load and filter RecipeNLG dataset"""

    print("Loading RecipeNLG dataset...")
    # RecipeNLG is usually in CSV or JSON format
    try:
        # Try CSV first
        df = pd.read_csv(file_path)
    except:
        try:
            # Try JSON
            df = pd.read_json(file_path)
        except:
            print("Error: Could not load dataset. Please check file format.")
            return None

    print(f"Loaded {len(df)} recipes")
    print(f"Columns: {df.columns.tolist()}")

    # Initialize filter
    filter_engine = RecipeCuisineFilter()

    # Filter recipes
    filtered_results = filter_engine.filter_recipes(df, target_per_cuisine)

    # Enhance Honduran dataset if needed
    if len(filtered_results['honduran']) < target_per_cuisine:
        additional_honduran = filter_engine.enhance_honduran_dataset(
            df, filtered_results)
        filtered_results['honduran'].extend(additional_honduran)

    # Print results
    for cuisine in ['mexican', 'indian', 'honduran']:
        count = len(filtered_results[cuisine])
        print(f"\n{cuisine.title()} recipes found: {count}")

        if count > 0:
            # Show confidence distribution
            confidence_counts = defaultdict(int)
            for recipe in filtered_results[cuisine]:
                confidence_counts[recipe['confidence']] += 1

            print(f"  Confidence levels: {dict(confidence_counts)}")

            # Show sample titles
            print("  Sample recipes:")
            for i, recipe in enumerate(filtered_results[cuisine][:3]):
                print(
                    f"    {i+1}. {recipe['title']} (score: {recipe['score']})")

    return filtered_results


def save_filtered_recipes(filtered_results: Dict, output_dir: str = "./filtered_recipes/"):
    """Save filtered recipes to separate files"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    for cuisine, recipes in filtered_results.items():
        output_file = f"{output_dir}/{cuisine}_recipes.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(recipes, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(recipes)} {cuisine} recipes to {output_file}")


# Example usage
if __name__ == "__main__":
    # Replace with your RecipeNLG file path
    file_path = "./archive/RecipeNLG_dataset.csv"  # or .json

    # Filter recipes
    results = load_and_filter_recipenlg(file_path, target_per_cuisine=1000)

    if results:
        # Save results
        save_filtered_recipes(results)

        # Create combined training dataset
        all_recipes = []
        for cuisine, recipes in results.items():
            all_recipes.extend(recipes)

        print(f"\nTotal filtered recipes: {len(all_recipes)}")
        print("Ready for model training!")

        # Save combined dataset
        with open("./filtered_recipes/combined_fusion_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(all_recipes, f, indent=2, ensure_ascii=False)

        print("Combined dataset saved to ./filtered_recipes/combined_fusion_dataset.json")
