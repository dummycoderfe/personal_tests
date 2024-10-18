import subprocess
import re

def run(func, m, n):
    # 执行命令并捕获输出
    try:
        command = ['/home/felix/composable_kernel/build/bin/ckProfiler', func, '0', '0', '0', '0', '1', '--length', str(m), str(n)]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        output_str = output.decode('utf-8')
        lines = output_str.split('\n')
        for line in lines:
            if 'best perf' in line:
                match = re.search(r'best perf = ([\d\.]+) ms', line)
                if match:
                    best_perf_value = float(match.group(1))
                    return best_perf_value
                else:
                    print("Failed to extract the performance value from the line.")
                    return None
        print("No line with 'best perf' found in the output.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e.output.decode('utf-8')}")
        return None