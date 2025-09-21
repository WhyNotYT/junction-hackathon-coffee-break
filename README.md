# JunctionX hackathon - Coffeebreak
# Aisti: Multi-Sensory Data Anomaly Detection Platform

**"Turning Data into Sights and Sounds"**

Aisti ("sensory" in Finnish) revolutionizes data cleaning by transforming datasets into visual and auditory representations to detect anomalies missed by traditional algorithms.

## Core Innovation
- **Multi-Modal Detection**: Combines statistical ML (Isolation Forest, LOF, OCSVM), computer vision morphological analysis, and audio frequency processing
- **Noise Kernel Method**: Creates noise samples from flagged data, applies PNR filtering, then identifies anomalies based on sample shift magnitude
- **Sensory Transformation**: Converts data correlations into playable frequencies and visual film strips for human pattern recognition

## Architecture
1. **LLM Supervisor**: Analyzes table headers/metadata to identify critical fields and reasonable value ranges
2. **Multi-Method Detection**: Parallel processing using 7+ anomaly detection algorithms with confidence voting
3. **Audio Analysis**: Data sonification reveals correlation patterns through frequency relationships
4. **Computer Vision**: Morphological operations on data-as-images detect structural anomalies
5. **Consensus Engine**: Flags entries detected by multiple processes with confidence scores
6. **Final LLM**: Generates actionable cleaning suggestions based on results

## Output
A dashboard displaying all intermediate steps, visualizations, audio playbacks, and cleaning recommendations with statistical confidence metrics.

**Result**: Automated, interpretable data cleaning that combines algorithmic precision with human-perceivable pattern recognition.