class PTYPE:
    header = '\033[95m'
    ok_blue = '\033[94m'
    ok_green = '\033[92m'
    warning = '\033[93m'
    fail = '\033[91m'
    end = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    none = ''


def f_print(s: str, ptype=PTYPE.none):
    print(ptype + s + PTYPE.end)


class Print:

    def __init__(self):
        self._content = ""
        self._stack_ptype = []

    def start(self, ptype):
        if ptype == PTYPE.none:
            raise("You can't start with 'PTYPE.end'.")
        self._content += ptype
        self._stack_ptype.append(ptype)

    def end(self):
        self._content += PTYPE.none
        self._stack_ptype.pop(-1)

    def write(self, s):
        self._content += s

    def flush(self):
        while len(self._stack_ptype) > 0:
            self.end()
        print(self._content)