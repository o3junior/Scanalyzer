import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import ipaddress
import warnings
warnings.filterwarnings('ignore')

class NmapScanDetector:
    def __init__(self, host_ip="192.168.29.238"):
        self.host_ip = host_ip
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(kernel='rbf', random_state=42)
        
    def load_wireshark_data(self, csv_path):
        """Load and preprocess Wireshark CSV data"""
        print("Loading Wireshark data...")
        
        # Try different encodings commonly used by Wireshark
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        df = None
        for encoding in encodings_to_try:
            try:
                print(f"Trying encoding: {encoding}")
                df = pd.read_csv(csv_path, low_memory=False, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                print(f"Failed with {encoding}: {str(e)[:100]}...")
                continue
            except Exception as e:
                print(f"Other error with {encoding}: {str(e)[:100]}...")
                continue
        
        if df is None:
            # Last resort: try with error handling
            try:
                print("Trying with error replacement...")
                df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8', errors='replace')
                print("Loaded with error replacement (some characters may be corrupted)")
            except Exception as e:
                raise Exception(f"Could not load file with any encoding. Error: {e}")
        
        # Clean column names (remove spaces and special characters)
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Show the actual columns we found
        print(f"Columns found: {list(df.columns)}")
        
        # Convert Time to datetime if it's not already (try different time formats)
        time_columns = [col for col in df.columns if 'time' in col.lower()]
        if time_columns:
            time_col = time_columns[0]
            print(f"Using time column: {time_col}")
            try:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                # Rename to standard 'Time' for consistency
                if time_col != 'Time':
                    df['Time'] = df[time_col]
            except:
                print("Could not parse time column")
        
        print(f"Loaded {len(df)} packets")
        print(f"Sample of first few rows:")
        print(df.head(2))
        
        return df
    
    def extract_port_from_info(self, info_str):
        """Extract port numbers from Info column"""
        if pd.isna(info_str) or not isinstance(info_str, str):
            return None
        
        # Common patterns for port extraction
        import re
        port_patterns = [
            r'→ (\d+)',  # Destination port
            r'Port (\d+)',
            r':(\d+)',
            r'DstPort=(\d+)'
        ]
        
        for pattern in port_patterns:
            match = re.search(pattern, info_str)
            if match:
                return int(match.group(1))
        return None
    
    def create_session_features(self, df, window_size=1000):
        """Create session-based features for scan detection using packet count windows"""
        print("Creating session-based features...")
        
        features_list = []
        
        # Sort by packet number instead of time since timestamps are all the same
        df = df.sort_values('No.')
        
        # Create windows based on packet count instead of time
        total_packets = len(df)
        num_windows = max(1, total_packets // window_size)
        
        print(f"Creating {num_windows} windows of ~{window_size} packets each")
        
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, total_packets)
            window_df = df.iloc[start_idx:end_idx]
            
            # Separate outgoing and incoming traffic
            outgoing = window_df[window_df['Source'] == self.host_ip]
            incoming = window_df[window_df['Destination'] == self.host_ip]
            
            if len(outgoing) == 0:
                continue
                
            features = {}
            
            # Basic counting features
            features['total_packets'] = len(window_df)
            features['outgoing_packets'] = len(outgoing)
            features['incoming_packets'] = len(incoming)
            features['packet_ratio'] = len(outgoing) / max(len(incoming), 1)
            
            # Destination diversity (key scan indicator)
            unique_destinations = outgoing['Destination'].nunique()
            features['unique_destinations'] = unique_destinations
            features['dest_diversity_ratio'] = unique_destinations / max(len(outgoing), 1)
            
            # Protocol diversity
            features['protocol_diversity'] = outgoing['Protocol'].nunique()
            features['tcp_packets'] = len(outgoing[outgoing['Protocol'] == 'TCP'])
            features['udp_packets'] = len(outgoing[outgoing['Protocol'] == 'UDP'])
            features['icmp_packets'] = len(outgoing[outgoing['Protocol'] == 'ICMP'])
            
            # Since we don't have reliable timestamps, use packet ordering for timing features
            features['avg_time_between_packets'] = 1  # Placeholder
            features['std_time_between_packets'] = 0  # Placeholder
            features['min_time_between_packets'] = 1  # Placeholder
            features['rapid_packets'] = len(outgoing)  # Assume all are rapid since timing is unreliable
            
            # Port scanning features
            if 'Info' in window_df.columns:
                ports = []
                for info in outgoing['Info']:
                    port = self.extract_port_from_info(info)
                    if port:
                        ports.append(port)
                
                if ports:
                    features['unique_ports'] = len(set(ports))
                    features['port_range'] = max(ports) - min(ports) if len(ports) > 1 else 0
                    features['sequential_ports'] = self.detect_sequential_ports(ports)
                    features['common_ports'] = self.count_common_ports(ports)
                else:
                    features['unique_ports'] = 0
                    features['port_range'] = 0
                    features['sequential_ports'] = 0
                    features['common_ports'] = 0
            
            # Packet size features
            if 'Length' in window_df.columns:
                lengths = pd.to_numeric(outgoing['Length'], errors='coerce').dropna()
                if len(lengths) > 0:
                    features['avg_packet_size'] = lengths.mean()
                    features['std_packet_size'] = lengths.std()
                    features['small_packets'] = (lengths < 100).sum()
                else:
                    features['avg_packet_size'] = 0
                    features['std_packet_size'] = 0
                    features['small_packets'] = 0
            
            # Response analysis
            features['response_ratio'] = len(incoming) / max(len(outgoing), 1)
            features['no_response_ratio'] = max(0, (len(outgoing) - len(incoming)) / max(len(outgoing), 1))
            
            # Window identifier and packet position
            features['window_id'] = i
            features['packet_position'] = start_idx  # Use packet position instead of time
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        print(f"Created {len(features_df)} feature windows")
        return features_df
    
    def analyze_traffic_patterns(self, df):
        """Analyze overall traffic patterns to understand the dataset"""
        print("\n--- Traffic Analysis ---")
        
        # Basic statistics
        total_packets = len(df)
        outgoing = df[df['Source'] == self.host_ip]
        incoming = df[df['Destination'] == self.host_ip]
        
        print(f"Total packets: {total_packets}")
        print(f"Outgoing packets: {len(outgoing)} ({len(outgoing)/total_packets*100:.1f}%)")
        print(f"Incoming packets: {len(incoming)} ({len(incoming)/total_packets*100:.1f}%)")
        
        # Protocol distribution
        print(f"\nProtocol Distribution:")
        protocol_counts = df['Protocol'].value_counts()
        for protocol, count in protocol_counts.head(10).items():
            print(f"  {protocol}: {count} ({count/total_packets*100:.1f}%)")
        
        # Top destinations
        print(f"\nTop Destinations from your host:")
        dest_counts = outgoing['Destination'].value_counts()
        for dest, count in dest_counts.head(10).items():
            print(f"  {dest}: {count} packets")
        
        # Time span analysis
        if 'Time' in df.columns:
            time_span = df['Time'].max() - df['Time'].min()
            print(f"\nCapture Duration: {time_span}")
            
        return {
            'total_packets': total_packets,
            'outgoing_ratio': len(outgoing)/total_packets,
            'protocol_dist': protocol_counts,
            'top_destinations': dest_counts.head(10)
        }
    
    def detect_sequential_ports(self, ports):
        """Detect if ports are accessed sequentially (scan indicator)"""
        if len(ports) < 3:
            return 0
        
        sorted_ports = sorted(set(ports))
        sequential_count = 0
        
        for i in range(len(sorted_ports) - 2):
            if sorted_ports[i+1] - sorted_ports[i] == 1 and sorted_ports[i+2] - sorted_ports[i+1] == 1:
                sequential_count += 1
        
        return sequential_count
    
    def count_common_ports(self, ports):
        """Count access to common service ports"""
        common_ports = {22, 23, 25, 53, 80, 110, 143, 443, 993, 995}
        return len(set(ports) & common_ports)
    
    def create_labels(self, features_df, scan_threshold=5):
        """Create labels based on scanning behavior patterns"""
        print("Creating labels...")
        
        # More refined heuristic labeling for mixed traffic
        labels = []
        
        for _, row in features_df.iterrows():
            score = 0
            
            # High destination diversity (key indicator)
            if row['unique_destinations'] > scan_threshold:
                score += 4
            elif row['unique_destinations'] > 2:
                score += 1
            
            # Port scanning behavior
            if row['unique_ports'] > 15:
                score += 3
            elif row['unique_ports'] > 5:
                score += 1
            
            # Rapid packet sending (aggressive scanning)
            if row['rapid_packets'] > 20:
                score += 3
            elif row['rapid_packets'] > 5:
                score += 1
            
            # Sequential port access (systematic scanning)
            if row['sequential_ports'] > 2:
                score += 3
            elif row['sequential_ports'] > 0:
                score += 1
            
            # Low response ratio (many failed connections)
            if row['response_ratio'] < 0.2:
                score += 2
            elif row['response_ratio'] < 0.5:
                score += 1
            
            # High outgoing packet ratio
            if row['packet_ratio'] > 10:
                score += 2
            elif row['packet_ratio'] > 3:
                score += 1
            
            # Large number of packets in short time
            if row['total_packets'] > 100:
                score += 1
            
            # Very fast packet intervals (aggressive scanning)
            if row['min_time_between_packets'] < 0.01 and row['min_time_between_packets'] > 0:
                score += 2
            
            # Label as scan if score >= 6 (more conservative for mixed traffic)
            labels.append(1 if score >= 6 else 0)
        
        return np.array(labels)
    
    def train_models(self, features_df, labels):
        """Train ML models for scan detection"""
        print("Training models...")
        
        # Select features for training (exclude metadata columns)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['window_id', 'packet_position']]
        X = features_df[feature_cols]
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        
        # Train SVM
        self.svm_model.fit(X_train_scaled, y_train)
        svm_pred = self.svm_model.predict(X_test_scaled)
        
        # Evaluate models
        print("\n--- Random Forest Results ---")
        print(classification_report(y_test, rf_pred))
        
        print("\n--- SVM Results ---")
        print(classification_report(y_test, svm_pred))
        
        return {
            'X_test': X_test,
            'y_test': y_test,
            'rf_pred': rf_pred,
            'svm_pred': svm_pred,
            'feature_names': feature_cols
        }
    
    def analyze_feature_importance(self, feature_names):
        """Analyze which features are most important for detection"""
        importance = self.rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\n--- Top 10 Most Important Features ---")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def visualize_results(self, results, features_df, labels):
        """Create visualizations of the detection results"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confusion Matrix for Random Forest
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(results['y_test'], results['rf_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 2: Feature Importance
        plt.subplot(2, 3, 2)
        importance = self.analyze_feature_importance(results['feature_names'])
        top_features = importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        
        # Plot 3: Scan Activity by Window Position (instead of time)
        plt.subplot(2, 3, 3)
        scan_timeline = features_df.copy()
        scan_timeline['is_scan'] = labels
        # Group by window position ranges
        window_ranges = pd.cut(scan_timeline['window_id'], bins=10)
        window_scans = scan_timeline.groupby(window_ranges)['is_scan'].sum()
        plt.plot(range(len(window_scans)), window_scans.values, marker='o')
        plt.title('Scan Activity by Window Position')
        plt.xlabel('Window Range')
        plt.ylabel('Number of Scans')
        plt.xticks(rotation=45)
        
        # Plot 4: Destination Diversity Distribution
        plt.subplot(2, 3, 4)
        normal_data = features_df[labels==0]['unique_destinations']
        scan_data = features_df[labels==1]['unique_destinations']
        
        if len(normal_data) > 0 and len(scan_data) > 0:
            plt.hist([normal_data, scan_data], 
                    bins=30, alpha=0.7, label=['Normal', 'Scan'])
        elif len(scan_data) > 0:
            plt.hist(scan_data, bins=30, alpha=0.7, label='Scan', color='orange')
        else:
            plt.hist(normal_data, bins=30, alpha=0.7, label='Normal', color='blue')
            
        plt.title('Destination Diversity Distribution')
        plt.xlabel('Unique Destinations')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 5: Packet Rate Distribution
        plt.subplot(2, 3, 5)
        normal_packets = features_df[labels==0]['total_packets']
        scan_packets = features_df[labels==1]['total_packets']
        
        if len(normal_packets) > 0 and len(scan_packets) > 0:
            plt.hist([normal_packets, scan_packets], 
                    bins=30, alpha=0.7, label=['Normal', 'Scan'])
        elif len(scan_packets) > 0:
            plt.hist(scan_packets, bins=30, alpha=0.7, label='Scan', color='orange')
        else:
            plt.hist(normal_packets, bins=30, alpha=0.7, label='Normal', color='blue')
            
        plt.title('Packet Rate Distribution')
        plt.xlabel('Packets per Window')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Plot 6: Response Ratio Distribution
        plt.subplot(2, 3, 6)
        normal_response = features_df[labels==0]['response_ratio']
        scan_response = features_df[labels==1]['response_ratio']
        
        if len(normal_response) > 0 and len(scan_response) > 0:
            plt.hist([normal_response, scan_response], 
                    bins=30, alpha=0.7, label=['Normal', 'Scan'])
        elif len(scan_response) > 0:
            plt.hist(scan_response, bins=30, alpha=0.7, label='Scan', color='orange')
        else:
            plt.hist(normal_response, bins=30, alpha=0.7, label='Normal', color='blue')
            
        plt.title('Response Ratio Distribution')
        plt.xlabel('Response Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def detect_scans(self, csv_path, window_size=1000):
        """Main method to detect scans from Wireshark CSV"""
        # Load data
        df = self.load_wireshark_data(csv_path)
        
        # Analyze overall traffic patterns first
        traffic_stats = self.analyze_traffic_patterns(df)
        
        # Create features using packet-count windows instead of time windows
        features_df = self.create_session_features(df, window_size)
        
        # Create labels
        labels = self.create_labels(features_df)
        
        print(f"\nDetected {labels.sum()} potential scan windows out of {len(labels)} total windows")
        print(f"Scan ratio: {labels.sum()/len(labels)*100:.2f}%")
        
        # Show scanning patterns
        if labels.sum() > 0:
            scan_features = features_df[labels == 1]
            print(f"\nScanning Pattern Summary:")
            print(f"  Avg destinations per scan window: {scan_features['unique_destinations'].mean():.1f}")
            print(f"  Avg ports per scan window: {scan_features['unique_ports'].mean():.1f}")
            print(f"  Avg packets per scan window: {scan_features['total_packets'].mean():.1f}")
        
        # Only train models if we have enough data points
        if len(features_df) > 10:
            # Train models
            results = self.train_models(features_df, labels)
            
            # Visualize results
            self.visualize_results(results, features_df, labels)
            
            return features_df, labels, results, traffic_stats
        else:
            print(f"Not enough windows ({len(features_df)}) for model training. Try reducing window_size.")
            return features_df, labels, None, traffic_stats

# Usage example
if __name__ == "__main__":
    # Initialize detector with your host IP
    detector = NmapScanDetector(host_ip="192.168.29.238")
    
    # Process your Wireshark CSV file
    csv_path = "traffic_final.csv"
    
    try:
        features_df, labels, results, traffic_stats = detector.detect_scans(csv_path, window_size=1000)
        
        # Additional analysis
        print("\n--- Scan Detection Summary ---")
        print(f"Total time windows analyzed: {len(features_df)}")
        print(f"Scan windows detected: {labels.sum()}")
        print(f"Normal windows: {len(labels) - labels.sum()}")
        
        # Show some detected scan windows
        if labels.sum() > 0:
            scan_windows = features_df[labels == 1].head()
            print("\n--- Sample Detected Scan Windows ---")
            print(scan_windows[['unique_destinations', 'unique_ports', 'rapid_packets', 'response_ratio']])
        
    except Exception as e:
        print(f"Error processing file: {e}")
        print("Make sure your CSV file path is correct and the file is accessible.")