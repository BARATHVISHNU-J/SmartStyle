import os
import django
import sys

# Setup Django
sys.path.append(os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'smartstyle.settings')
django.setup()

from chat.models import FashionItem, Trend

def populate_fashion_data():
    """Populate initial fashion data"""

    # Clear existing data
    FashionItem.objects.all().delete()
    Trend.objects.all().delete()

    # Fashion items
    fashion_items = [
        {
            'name': 'Classic White Button-Down Shirt',
            'category': 'tops',
            'description': 'A versatile white button-down shirt perfect for business casual or formal occasions. Made from premium cotton with a tailored fit.',
            'season': 'all',
            'occasion': 'business, formal, casual',
            'gender': 'unisex',
            'price_range': 'mid-range',
            'brands': ['Buttoned Down', 'Classic Fit', 'Premium Cotton'],
            'colors': ['white', 'light blue', 'gray']
        },
        {
            'name': 'Slim Fit Chinos',
            'category': 'bottoms',
            'description': 'Comfortable slim-fit chinos in neutral colors. Perfect for office wear or casual outings.',
            'season': 'spring, fall',
            'occasion': 'business, casual',
            'gender': 'unisex',
            'price_range': 'mid-range',
            'brands': ['Comfort Wear', 'Office Ready', 'Casual Classics'],
            'colors': ['khaki', 'navy', 'gray', 'black']
        },
        {
            'name': 'Leather Loafers',
            'category': 'shoes',
            'description': 'Classic leather loafers that transition from office to casual wear seamlessly.',
            'season': 'fall, winter',
            'occasion': 'business, formal',
            'gender': 'unisex',
            'price_range': 'premium',
            'brands': ['Leather Luxe', 'Classic Footwear', 'Executive Style'],
            'colors': ['black', 'brown', 'tan']
        },
        {
            'name': 'Floral Summer Dress',
            'category': 'dresses',
            'description': 'Light and airy floral print dress perfect for summer occasions and casual outings.',
            'season': 'summer',
            'occasion': 'casual, date, party',
            'gender': 'female',
            'price_range': 'budget',
            'brands': ['Summer Bloom', 'Floral Fashion', 'Casual Chic'],
            'colors': ['floral patterns', 'pastels']
        },
        {
            'name': 'Athletic Performance T-Shirt',
            'category': 'tops',
            'description': 'Moisture-wicking athletic t-shirt designed for workouts and sports activities.',
            'season': 'all',
            'occasion': 'athletic',
            'gender': 'unisex',
            'price_range': 'budget',
            'brands': ['Sport Fit', 'Active Wear', 'Performance Gear'],
            'colors': ['black', 'navy', 'gray', 'white']
        },
        {
            'name': 'Wool Overcoat',
            'category': 'outerwear',
            'description': 'Warm wool overcoat perfect for winter weather and formal occasions.',
            'season': 'winter',
            'occasion': 'formal, business',
            'gender': 'unisex',
            'price_range': 'premium',
            'brands': ['Winter Warmth', 'Executive Outerwear', 'Luxury Wool'],
            'colors': ['black', 'navy', 'gray', 'camel']
        },
        {
            'name': 'Denim Jacket',
            'category': 'outerwear',
            'description': 'Classic denim jacket that works for casual and transitional weather.',
            'season': 'spring, fall',
            'occasion': 'casual',
            'gender': 'unisex',
            'price_range': 'mid-range',
            'brands': ['Denim Co', 'Casual Wear', 'Vintage Style'],
            'colors': ['blue denim', 'black denim', 'light wash']
        },
        {
            'name': 'Evening Gown',
            'category': 'dresses',
            'description': 'Elegant evening gown for formal events and special occasions.',
            'season': 'fall, winter',
            'occasion': 'formal, party',
            'gender': 'female',
            'price_range': 'premium',
            'brands': ['Evening Elegance', 'Formal Wear', 'Designer Gowns'],
            'colors': ['black', 'navy', 'emerald', 'burgundy']
        },
        {
            'name': 'Running Sneakers',
            'category': 'shoes',
            'description': 'High-performance running sneakers with cushioning and support.',
            'season': 'all',
            'occasion': 'athletic',
            'gender': 'unisex',
            'price_range': 'mid-range',
            'brands': ['Run Fast', 'Athletic Gear', 'Performance Shoes'],
            'colors': ['black', 'white', 'neon']
        },
        {
            'name': 'Cashmere Sweater',
            'category': 'tops',
            'description': 'Luxurious cashmere sweater perfect for layering and cold weather.',
            'season': 'fall, winter',
            'occasion': 'casual, business',
            'gender': 'unisex',
            'price_range': 'premium',
            'brands': ['Cashmere Comfort', 'Luxury Knits', 'Winter Warmth'],
            'colors': ['cream', 'navy', 'gray', 'burgundy']
        }
    ]

    # Trends
    trends = [
        {
            'title': 'Sustainable Fashion',
            'description': 'Eco-friendly materials and ethical production practices are becoming mainstream.',
            'season': 'all',
            'year': 2024,
            'categories': ['tops', 'bottoms', 'outerwear']
        },
        {
            'title': 'Oversized Silhouettes',
            'description': 'Comfortable, oversized fits in casual wear and athleisure.',
            'season': 'all',
            'year': 2024,
            'categories': ['tops', 'bottoms', 'outerwear']
        },
        {
            'title': 'Bold Color Combinations',
            'description': 'Unexpected color pairings and vibrant hues in accessories and outfits.',
            'season': 'spring, summer',
            'year': 2024,
            'categories': ['tops', 'accessories', 'shoes']
        },
        {
            'title': 'Tech-Integrated Clothing',
            'description': 'Garments with built-in technology for fitness tracking and temperature regulation.',
            'season': 'all',
            'year': 2024,
            'categories': ['tops', 'bottoms', 'outerwear']
        }
    ]

    # Create fashion items
    for item_data in fashion_items:
        FashionItem.objects.create(**item_data)

    # Create trends
    for trend_data in trends:
        Trend.objects.create(**trend_data)

    print(f"Created {len(fashion_items)} fashion items and {len(trends)} trends")

if __name__ == '__main__':
    populate_fashion_data()
