@register_task("technical_debt")
class TechnicalDebtTask(BaseTaskModule):

    TASK_NAME   = "technical_debt"
    INPUT_LABEL = "Input (Java code)"

    def load_dataset(self, path: str, fmt: str) -> list[dict]:
        # Use sep=';' because your file is semicolon-delimited
        df = pd.read_csv(path, sep=';', quotechar='"')

        code_col  = self._find_column(df.columns, ["code_snippet", "code", "snippet", "source"])
        label_col = self._find_column(df.columns, ["smell", "label", "code_smell", "type"])

        dataset = []
        for _, row in df.iterrows():
            if pd.isna(row[code_col]) or pd.isna(row[label_col]):
                continue
            
            # CRITICAL FIX: Map 'blob' from the CSV to 'god_class' used in prompts
            raw_label = str(row[label_col]).strip().lower().replace(" ", "_").replace("-", "_")
            if raw_label == "blob":
                raw_label = "god_class"
            
            dataset.append({"code": str(row[code_col]), "label": raw_label})

        print(f"   Loaded {len(dataset)} samples. 'blob' mapped to 'god_class'.")
        return dataset

    def parse_response(self, raw_text: str) -> str:
        """Improved parsing to catch the model's output more reliably."""
        text = raw_text.strip().lower()

        # Check for specific smells first
        if any(x in text for x in ["god_class", "god class", "blob"]):
            return "god_class"
        if any(x in text for x in ["long_method", "long method"]):
            return "long_method"
        if any(x in text for x in ["data_class", "data class"]):
            return "data_class"
        if any(x in text for x in ["feature_envy", "feature envy"]):
            return "feature_envy"
        
        # Default to none
        return "none"