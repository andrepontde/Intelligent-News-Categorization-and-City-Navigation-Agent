# Intelligent News Categorization and City Navigation Agent

A comprehensive AI system that combines **news categorization** using machine learning with **intelligent city navigation** algorithms. This project demonstrates the integration of natural language processing and pathfinding algorithms for real-world applications.

## 🌟 Key Features

### 📰 News Categorization System
- **Naive Bayes Classification**: Automatically categorizes news articles using machine learning
- **RDF Ontology Integration**: Semantic data processing with XML-based ontology mapping
- **Multi-Model Support**: Comprehensive text classification using different features
- **Performance Metrics**: Detailed accuracy reports and confusion matrix analysis

### 🗺️ City Navigation & Route Optimization
- **A\* Search Algorithm**: Optimal pathfinding between cities with heuristic optimization
- **Uniform Cost Search**: Alternative pathfinding for guaranteed optimal routes
- **Traveling Salesman Problem**: Genetic algorithm solution for multi-city route optimization
- **Interactive City Selection**: User-friendly interface for choosing destinations

## 🏗️ Project Structure

```
├── 📊 News Categorization
│   ├── Bayes_News_Categorizers/
│   │   ├── News_Models.py           # Core ML models for text classification
│   │   ├── Ontology_Interpreter.py  # RDF/XML ontology processing
│   │   ├── S&L_NBM.ipynb           # Model testing & validation
│   │   └── parsed_news_data.csv     # Processed training data
│   │
├── 🧭 Navigation Algorithms
│   ├── Path_Finder_Algorithms/
│   │   ├── A_star_city_search.py    # A* pathfinding implementation
│   │   ├── UC_city_search.py        # Uniform Cost Search algorithm
│   │   ├── CityMaps.py             # City network graph data
│   │   └── Search_testing.ipynb     # Algorithm testing & comparison
│   │
├── 🚗 Route Optimization
│   ├── Traveling_Salesman_Algorithm/
│   │   ├── TSP_with_GA.py          # Genetic algorithm for TSP
│   │   └── tsp_w_ga_results.ipynb  # Results visualization
│   │
├── 📋 Main Notebooks
│   ├── CityPulse_AI.ipynb          # Complete system integration
│   ├── CityMap.ipynb               # City network visualization
│   │
└── 📄 Data Files
    ├── Daily_News.csv              # Raw news dataset
    └── News_Categorizer_RDF.xml    # RDF ontology definitions
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### Quick Start

#### 1. News Categorization
```python
from Bayes_News_Categorizers.News_Models import *

# Load and explore the dataset
print_dataset_shape()

# Train and test different classification models
short_description_classifier()
headline_classifier() 
combined_features_classifier()
```

#### 2. City Navigation
```python
from Path_Finder_Algorithms.A_star_city_search import a_star_search
from Path_Finder_Algorithms.CityMaps import city_graph, heuristics

# Find optimal route between two cities
path, cost = a_star_search(city_graph, "Phoenix", "Los Angeles", heuristics)
print(f"Optimal route: {path}")
print(f"Total distance: {cost} miles")
```

#### 3. Multi-City Route Optimization
```python
from Traveling_Salesman_Algorithm.TSP_with_GA import *

# Interactive city selection and route optimization
select_cities_to_visit()  # Choose cities to visit
# Genetic algorithm finds optimal tour starting/ending at Phoenix
```

## 📊 Sample Results

### News Classification Performance
- **Short Description Model**: ~85% accuracy on news categorization
- **Headline Model**: ~78% accuracy with confusion matrix analysis
- **Combined Features**: Enhanced performance using multiple text features

### Navigation Algorithms
- **A\* Search**: Optimal pathfinding with heuristic guidance
- **Uniform Cost**: Guaranteed optimal solution exploration
- **TSP with GA**: Multi-city tour optimization starting from Phoenix

## 🎯 Use Cases

1. **Media Organizations**: Automated news categorization for content management
2. **Logistics Companies**: Optimal route planning for delivery services
3. **Travel Planning**: Multi-destination trip optimization
4. **Research**: Comparative analysis of search algorithms and ML models

## 🔬 Technical Details

### News Categorization
- **Algorithm**: Multinomial Naive Bayes with CountVectorizer
- **Features**: Text preprocessing, stop word removal, feature selection
- **Evaluation**: Cross-validation, classification reports, visualization

### Navigation Algorithms
- **A\* Search**: Uses Manhattan/Euclidean distance heuristics
- **TSP Genetic Algorithm**: Population-based optimization with mutation/crossover
- **Graph Representation**: Weighted adjacency list for city networks

## 📝 Testing & Validation

- **Jupyter Notebooks**: Interactive testing environments for all components
- **Model Persistence**: Save/load functionality for trained models
- **Performance Metrics**: Comprehensive evaluation with visualizations
- **Algorithm Comparison**: Side-by-side testing of different approaches

## 👨‍💻 Author
Andre Pont - 23164034

---

**Note**: For detailed testing and usage examples, refer to the individual Jupyter notebooks in each module directory.

(Readme made with AI)