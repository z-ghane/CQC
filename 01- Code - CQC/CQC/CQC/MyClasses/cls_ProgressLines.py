import enlighten



# IV. ProgressLines

class cls_ProgressLines():
    
    def progress_lines(self, num, total, description, unit, colour):
        desc = self.set_strings_to_equal_len_(description)
        manager = enlighten.get_manager()
        progresses = []
        for i in range(num):
            prog = manager.counter(total=total[i], desc=desc[i], unit=unit[i], color=colour[i])
            prog.refresh()
            progresses.append(prog)
        self.progresses = progresses
    
    
    def set_strings_to_equal_len_(self, description):
        max_len = 0
        # longest_string_length = len(max(description, key=len))
        longest_string_length = -1
        for ele in description:
            if len(ele) > longest_string_length:
                longest_string_length = len(ele)
        w = []
        for i, word in enumerate(description):
            temp = longest_string_length - len(word)
            w.append(word + " " * temp)

        return w







