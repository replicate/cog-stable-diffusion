strace -f -e trace=file -o >(ts -s '%.s' > file_trace) python3 -m cog.server.http
rg -v ENOENT file_trace|rg -v getcwd|rg '\w+\('|rg '^(\d|\[)'|rg -or '$1 $2' '^(\d+\.\d+).*?"(.*?)"'|awk '!seen[$2]++ {print $1, $2}'|xargs -L1 bash -c 'if [ -f "$1" ] && [ ! -z "${1// /}" ]; then echo "$0 $(realpath $1)"; fi'

