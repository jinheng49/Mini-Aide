import random
from typing import Callable, Optional
from .journal import Node, Journal, MetricValue
from .llm_backend import llm_generate_code

class MiniAIDECore:
    def __init__(self, task_desc: str, eval_metric: str):
        # ✅ 无LLM配置，自动读取全局统一配置
        self.task_desc = task_desc
        self.eval_metric = eval_metric
        self.journal = Journal()

    # 原子算子1：Draft
    def draft(self) -> Node:
        prompt = self._build_draft_prompt()
        plan, code = llm_generate_code(prompt)  # 无参数，统一调用
        node = Node(code=code, plan=plan)
        self.journal.add_node(node)
        return node

    # 原子算子2：Improve
    def improve(self, parent_node: Node) -> Node:
        prompt = self._build_improve_prompt(parent_node)
        plan, code = llm_generate_code(prompt)
        node = Node(code=code, plan=plan)
        parent_node.add_child(node)
        self.journal.add_node(node)
        return node

    # 原子算子3：Debug
    def debug(self, buggy_node: Node) -> Node:
        prompt = self._build_debug_prompt(buggy_node)
        plan, code = llm_generate_code(prompt)
        node = Node(code=code, plan=plan, debug_depth=buggy_node.debug_depth + 1)
        buggy_node.add_child(node)
        self.journal.add_node(node)
        return node

    # 启发式节点选择（不变）
    def select_next_node(self) -> Optional[Node]:
        debuggable = [n for n in self.journal.buggy_nodes if n.debug_depth < 3]
        if debuggable:
            return random.choice(debuggable)
        best = self.journal.get_best_node()
        if best:
            return best
        return None

    # Prompt 构建
    def _build_draft_prompt(self) -> str:
        return f"""
        Generate COMPLETE, ERROR-FREE Python code for the following task:

        Task: {self.task_desc}
        Evaluation Metric: {self.eval_metric}

        CRITICAL REQUIREMENTS:
        1. Read data from './data/' directory (train.csv, test.csv) - NOT './input/'
        2. Implement proper data preprocessing and feature engineering
        3. Use appropriate machine learning models
        4. MUST use train_test_split for model evaluation
        5. Calculate and print the evaluation metric (accuracy)
        6. CRITICAL: Output format MUST be exactly: print("Metric: <value>") where <value> is a float between 0 and 1
        7. Ensure ALL parentheses, brackets, and quotes are properly closed
        8. Code must be syntactically correct and runnable without errors
        9. Include proper imports at the top
        10. Use if __name__ == "__main__": guard for main execution
        11. Add print statements for debugging and progress tracking
        12. CRITICAL: After evaluation, generate submission.csv file in current directory

        CODE STRUCTURE (STRICT - Follow this exactly):
        ```python
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        # ... other imports

        def main():
            # Load data from './data/'
            train_df = pd.read_csv('./data/train.csv')
            test_df = pd.read_csv('./data/test.csv')
            
            # Preprocess
            X = train_df.drop('label', axis=1).values / 255.0
            y = train_df['label'].values
            
            # Split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = YourModel()
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Metric: {{accuracy}}")
            
            # Generate submission
            test_predictions = model.predict(test_df.values / 255.0)
            submission = pd.DataFrame({{
                'ImageId': range(1, len(test_predictions) + 1),
                'Label': test_predictions
            }})
            submission.to_csv('submission.csv', index=False)
            print("Submission saved!")

        if __name__ == "__main__":
            main()
        ```

        DIVERSITY GUIDELINES:
        - Try DIFFERENT machine learning models each time (RandomForest, SVM, XGBoost, Neural Networks, etc.)
        - Experiment with different feature engineering approaches
        - Vary hyperparameters and model architectures
        - Consider ensemble methods and advanced techniques
        - Be creative with preprocessing strategies
        - Each iteration should explore a unique approach

        FAILURE CONDITIONS:
        - Missing or incorrect output format
        - Syntax errors (unclosed brackets, quotes, etc.)
        - Code that crashes or hangs
        - Missing metric output
        - Missing submission.csv generation
        - Using wrong data directory (must be './data/', not './input/')
        - Code starts with text explanations instead of import statements
        - Using markdown formatting in code

        OUTPUT FORMAT:
        - First line: Brief explanation of your approach (1-2 sentences)
        - Second line: Empty line
        - Rest: Complete Python code starting with import statements
        - NO markdown formatting in code
        - NO text explanations mixed with code
        - NO numbered lists or bullet points in code

        Provide ONLY the brief explanation followed by complete Python code.
        """

    def _build_improve_prompt(self, parent_node: Node) -> str:
        return f"""
        IMPROVE the following code to achieve better {self.eval_metric}:
        
        Current Best Metric: {self.journal.get_best_node().metric.value if self.journal.get_best_node() else 'N/A'}
        
        Original Code:
        {parent_node.code}
        
        IMPROVEMENT REQUIREMENTS:
        1. Maintain the core logic and structure
        2. Enhance feature engineering, model selection, or hyperparameter tuning
        3. Ensure ALL syntax is correct with properly closed brackets and quotes
        4. CRITICAL: Output format MUST be exactly: print("Metric: <value>")
        5. Optimize for higher accuracy
        6. Code must be runnable without errors
        7. Include progress tracking print statements
        8. CRITICAL: Generate submission.csv file after evaluation
        9. CRITICAL: Read data from './data/' directory (NOT './input/')
        
        CODE STRUCTURE (STRICT - Follow this exactly):
        - Use simple structure without nested try-except blocks
        - Load data directly from './data/'
        - Train model
        - Print metric: print(f"Metric: {{accuracy}}")
        - Generate submission.csv
        
        FAILURE CONDITIONS:
        - Syntax errors or unclosed brackets
        - Missing or incorrect metric output format
        - Code that crashes
        - Degraded performance
        - Missing submission.csv generation
        - Using wrong data directory (must be './data/', not './input/')
        - Code starts with text explanations instead of import statements
        - Using markdown formatting in code
        
        OUTPUT FORMAT:
        - First line: Brief explanation of improvements (1-2 sentences)
        - Second line: Empty line
        - Rest: Complete Python code starting with import statements
        - NO markdown formatting in code
        - NO text explanations mixed with code
        - NO numbered lists or bullet points in code
        
        Provide ONLY the brief explanation followed by improved Python code.
        """

    def _build_debug_prompt(self, buggy_node: Node) -> str:
        return f"""
        DEBUG and FIX the following buggy code:
        
        Original Code:
        {buggy_node.code}
        
        DEBUGGING REQUIREMENTS:
        1. Fix ALL syntax errors (unclosed brackets, quotes, parentheses, etc.)
        2. Fix ALL runtime errors
        3. Ensure all function calls are correct
        4. CRITICAL: Output format MUST be exactly: print("Metric: <value>")
        5. Maintain the original modeling approach and algorithm logic
        6. Ensure code is runnable without errors
        7. Add print statements for debugging
        8. Verify all imports are correct
        9. Check for common issues: variable scope, data types, file paths
        10. CRITICAL: Generate submission.csv file after evaluation
        11. CRITICAL: Read data from './data/' directory (NOT './input/')
        
        CRITICAL CHECKLIST:
        - All brackets [ ], parentheses ( ), and quotes " " or ' ' are properly closed
        - All if statements have proper indentation
        - All function definitions have colons and proper indentation
        - File paths are correct ('./data/train.csv', NOT './input/train.csv')
        - Metric output is in exact format: print("Metric: 0.XX")
        - submission.csv is generated in current directory
        - NO try-except blocks unless absolutely necessary
        - Keep code structure simple and linear
        - Code starts with import statements, not text explanations
        - NO markdown formatting in code
        
        OUTPUT FORMAT:
        - First line: Brief explanation of fixes (1-2 sentences)
        - Second line: Empty line
        - Rest: Complete Python code starting with import statements
        - NO markdown formatting in code
        - NO text explanations mixed with code
        - NO numbered lists or bullet points in code
        
        Provide ONLY the brief explanation followed by corrected Python code.
        """
