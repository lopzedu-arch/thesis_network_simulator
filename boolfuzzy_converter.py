import re
import os
import tempfile
class BooleanToFuzzyProbabilistic:
    def __init__(self):
        self.ref_q = {}
        self.ref_p = {}
        self.log_dir = "./logs_conversion"
        os.makedirs(self.log_dir, exist_ok=True)
    def process_file(self, file=None, file_nodes=None, file_inputs=None):
        if None in (file, file_nodes, file_inputs):
            return None, "No file"
        try:
            fd_probabilistic, path_probabilistic = tempfile.mkstemp(suffix='.txt', prefix='probabilistic_')
            with open(file.name, 'r', encoding='utf-8') as f_in, \
                 os.fdopen(fd_probabilistic, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(self.convert_line(line) + '\n')
            return path_probabilistic, 'Success'
        except Exception as e:
            return None, f"Error: {str(e)}"
    def convert_boolean_expression(self, expr):
        if '!' in expr or '|' in expr or '&' in expr:
            if match := re.search(r'!(\w+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(f'!{match.group(1)}', f'(1-{match.group(1)})')
                )
            if match := re.search(r'([^&|]+)&([^&|]+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(match.group(0), f'{match.group(1)}*{match.group(2)}')
                )
            if match := re.search(r'([^&|]+)\|([^&|]+)', expr):
                return self.convert_boolean_expression(
                    expr.replace(match.group(0), f'({match.group(1)}+{match.group(2)}-{match.group(1)}*{match.group(2)})')
                )
        return expr
    def convert_line(self, boolean_expression):
        boolean_expression = boolean_expression.replace(' ', '')
        if '(' not in boolean_expression:
            return self.convert_boolean_expression(boolean_expression)
        markers = {}
        i = 0
        while '(' in boolean_expression:
            if match := re.search(r'\(([^()]+)\)', boolean_expression):
                marker = self.marker(i)
                markers[marker] = match.group(0)
                boolean_expression = boolean_expression.replace(match.group(0), marker)
                i += 1
            else:
                break
        boolean_expression = self.convert_boolean_expression(boolean_expression)
        for marker, parenthesis_expr in reversed(markers.items()):
            inner = parenthesis_expr.strip('()')
            converted_inner = self.convert_boolean_expression(inner)
            boolean_expression = boolean_expression.replace(marker, converted_inner)

        return boolean_expression
    def marker(self,marker_number :int=0) -> str:
        return f"_A{marker_number}_"
