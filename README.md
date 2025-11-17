# Computer Vision Assignment - Integral Images & Texture Classification

**Author:** Tanushree  
**Date:** October 2025  
**Course:** Computer Vision  

## Overview

This repository contains a comprehensive computer vision assignment implementing two main components:

1. **Integral Images & Haar Features** - Fast computation of rectangular features for object detection
2. **Texture Classification** - Comparing different feature extraction methods for texture recognition

## 🗂️ Repository Structure

```
CV assigment 2/
├── cv_assignment.ipynb     # Main Jupyter notebook with all implementations
├── README.md              # This documentation file
├── data/                  # Directory for images (place cameraman.jpg here)
└── kth-dataset/          # KTH-TIPS texture dataset directory
```

## 📋 Requirements

### Dependencies
- Python 3.7+
- NumPy
- Matplotlib
- SciPy
- Scikit-learn (optional, with pandas conflict fixes)

### Dataset Requirements
- **Cameraman image** (`cameraman.jpg` or `cameraman.png`) in `data/` folder
- **KTH-TIPS dataset** in `kth-dataset/` folder with texture subfolders

## 🚀 Quick Start

1. **Clone/Download** this repository
2. **Place required images:**
   ```
   data/cameraman.jpg          # For Haar feature demonstration
   kth-dataset/cotton/         # Texture class folders
   kth-dataset/sandpaper/      # with image files inside
   kth-dataset/...             # (10 texture classes total)
   ```
3. **Open** `cv_assignment.ipynb` in Jupyter/VSCode
4. **Run cells sequentially** - each section is self-contained

## 📚 Part 1: Integral Images & Haar Features

### What it does:
- Implements **integral image computation** from scratch
- Creates **fast rectangle sum queries** using 4-corner formula
- Demonstrates **3 types of Haar features**:
  - Vertical Edge (left-right contrast)
  - Horizontal Edge (top-bottom contrast) 
  - Horizontal Line (bright-dark-bright pattern)

### Key Functions:
```python
compute_integral_image(image)           # Creates integral image
get_rectangle_sum(integral, x1,y1,x2,y2) # Fast rectangle sum
haar_vertical_edge(integral, ...)       # Vertical edge detection
haar_horizontal_edge(integral, ...)     # Horizontal edge detection
haar_horizontal_line(integral, ...)     # Line pattern detection
```

### Applications:
- **Face detection** (Viola-Jones algorithm foundation)
- **Object recognition** using rectangular features
- **Real-time processing** due to O(1) rectangle queries

## 📊 Part 2: Texture Classification

### Methods Implemented:

#### 1. Raw Pixels
- **Approach:** Direct pixel intensities as features
- **Pros:** Simple, fast
- **Cons:** Sensitive to lighting, high dimensionality

#### 2. Local Binary Patterns (LBP)
- **Approach:** Compare each pixel with 8 neighbors → binary code → histogram
- **Pros:** Rotation invariant, compact features
- **Cons:** Limited to local texture patterns

#### 3. Bag of Visual Words (BoW)
- **Approach:** Extract patches → cluster into "visual words" → histogram of word frequencies
- **Pros:** Captures local structure, scale invariant
- **Cons:** Requires clustering, loses spatial information

#### 4. Histogram of Oriented Gradients (HoG)
- **Approach:** Compute gradients → bin by orientation → normalize in blocks
- **Pros:** Captures shape/edge information, robust to lighting
- **Cons:** Sensitive to orientation changes

### Implementation Approaches:

| Implementation | Classifier | Library Used | Pros | Cons |
|----------------|------------|--------------|------|------|
| **Custom** | KNN/Linear | Pure NumPy | No dependencies, educational | May be slower |
| **Built-in** | KNN | SciPy functions | Optimized, reliable | Limited algorithms |
| **Sklearn** | SVM | Scikit-learn | Professional grade | Pandas conflicts |

## 🔧 Technical Details

### Pandas Conflict Resolution
The notebook includes multiple strategies to handle sklearn-pandas conflicts:
- Environment variable fixes
- Module cache clearing  
- Pandas reinstallation
- Fallback to custom implementations

### Performance Optimizations
- **Vectorized operations** using NumPy
- **Efficient distance calculations** with SciPy cdist
- **Memory-efficient** patch extraction
- **Progress indicators** for long operations

### Feature Scaling
All methods include proper feature normalization:
```python
scaler = SimpleScaler()                    # Custom implementation
X_scaled = scaler.fit_transform(X_train)   # Normalize features
```

## 📈 Results & Analysis

The notebook provides comprehensive comparisons:

### Accuracy Comparison
- **Best overall:** Usually HoG or LBP features
- **Fastest:** Raw pixels with simple classifiers
- **Most robust:** LBP for texture-specific tasks

### Implementation Comparison
- **Custom vs Built-in vs Sklearn** across all methods
- **Speed vs Accuracy** trade-offs
- **Memory usage** analysis

### Detailed Metrics
Each method reports:
- Classification accuracy (%)
- Training time (seconds)
- Prediction time (seconds)
- Feature dimensionality

## 🎯 Key Learning Outcomes

1. **Integral Images:**
   - Understand cumulative sum computation
   - Learn O(1) rectangle queries
   - Implement Haar feature detection

2. **Feature Engineering:**
   - Compare different texture descriptors
   - Understand trade-offs between methods
   - Experience with real-world datasets

3. **Implementation Skills:**
   - Write algorithms from scratch
   - Handle library conflicts professionally
   - Create robust, well-documented code

## 🔍 Code Organization

### Cell Structure:
1. **Setup & Imports** (Cells 1-4)
2. **Integral Images** (Cells 5-10) 
3. **Haar Features** (Cells 11-14)
4. **Dataset Loading** (Cells 15-17)
5. **Classification Methods** (Cells 18-22)
6. **Comparisons** (Cells 23-26)

### Error Handling:
- Robust file path detection
- Graceful fallbacks for missing dependencies
- Clear error messages with solutions

### Documentation:
- Extensive inline comments
- Markdown explanations between code sections
- Real-world context and applications

## 🛠️ Troubleshooting

### Common Issues:

**1. Dataset not found:**
```
Solution: Place kth-dataset/ folder in correct location
Check: Ensure texture subfolders contain .jpg/.png files
```

**2. Sklearn-pandas conflicts:**
```
Solution: Run the pandas fix cell (Cell 25)
Alternative: Use only custom implementations
```

**3. Memory issues:**
```
Solution: Reduce max_patches_per_image parameter
Alternative: Use smaller image sizes
```

**4. Slow performance:**
```
Solution: Use built-in scipy functions
Alternative: Reduce dataset size for testing
```

## 📖 References & Further Reading

### Algorithms:
- **Viola-Jones Face Detection:** Original integral image paper
- **LBP:** Ojala et al. multiresolution texture analysis
- **HoG:** Dalal & Triggs pedestrian detection
- **BoW:** Sivic & Zisserman visual words

### Datasets:
- **KTH-TIPS:** Real texture database
- **Standard test images:** Cameraman, Lena, etc.

### Libraries:
- **NumPy:** Numerical computing
- **SciPy:** Scientific computing functions  
- **Matplotlib:** Visualization
- **Scikit-learn:** Machine learning algorithms

## 🎉 Conclusion

This assignment demonstrates:
- **Low-level computer vision** algorithm implementation
- **Multiple approaches** to the same problem
- **Performance analysis** and comparison
- **Real-world applicability** of academic concepts

The comprehensive comparison shows that **no single method is universally best** - the choice depends on:
- Speed requirements
- Accuracy needs  
- Available computational resources
- Specific application domain

**Next steps** could include:
- Combining multiple features
- Deep learning approaches
- Real-time implementation
- Mobile/embedded deployment

---

*This assignment provides a solid foundation in classical computer vision techniques while maintaining practical relevance for modern applications.*
