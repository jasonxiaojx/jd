def answer(l):
    a = list_of_list(l)
    return spit_out(sort(a))
    # your code here

def spit_out(l):
    s = []
    for i in l:
        s.append(spit(i))
    return s
def spit(l):
    s = ''
    for i in l:
        s += str(i) + '.'
    return s[0:len(s)-1]

def sort(l):
    while not ck(l):
        for i in range(len(l)-1):
            if later_version(l[i],l[i+1]):
                l[i], l[i+1] = l[i+1], l[i]
    return l

def ck(l):
    for i in range(len(l)-1):
        if later_version(l[i], l[i+1]):
            return False
    return True
#next_num returns the next number before the period.
def list_of_list(l):
    c = []
    for i in l:
        c.append(version_to_list(i))
    return c

def version_to_list(s):
    return s.split('.')
#returns true if l1 is a later version compared to l2:
def later_version(l1, l2):
    for i in range(min(len(l1),len(l2))):
        if int(l1[i]) > int(l2[i]):
            return True
        elif int(l1[i]) < int(l2[i]):
            return False
        else:
            continue
    if len(l1) > len(l2):
        return True
