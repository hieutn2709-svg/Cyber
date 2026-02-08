import numpy as np
from typing import List, Dict, Set

class PerformanceMetrics:

    @staticmethod
    def calculate_precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

    def evaluate_entities(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:

        tp = 0
        fp = 0
        
        gt_set = {(e['start'], e['end'], e['type']) for e in ground_truth}
        pred_set = {(e['start'], e['end'], e['type']) for e in predictions}
        
        for pred in pred_set:
            if pred in gt_set:
                tp += 1
            else:
                fp += 1
        
        fn = len(gt_set) - tp
        
        return self.calculate_precision_recall_f1(tp, fp, fn)

    def evaluate_relations(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        tp = 0
        fp = 0
        
        gt_set = {(r['source'], r['target'], r['relationship']) for r in ground_truth}
        pred_set = {(r['source'], r['target'], r['relationship']) for r in predictions}
        
        for pred in pred_set:
            if pred in gt_set:
                tp += 1
            else:
                fp += 1
                
        fn = len(gt_set) - tp
        
        return self.calculate_precision_recall_f1(tp, fp, fn)

if __name__ == "__main__":
    metrics = PerformanceMetrics()
    
