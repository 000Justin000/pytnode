taskset --cpu-list 0 python tnode.py --dataset exponential_hawkes --niters 2000 --jump_type read --batch_size 30 --nsave 20 --evnt_align --restart > exponential_hawkes/7612c40.log 2>exponential_hawkes/7612c40.error &
taskset --cpu-list 1 python tnode.py --dataset self_inhibiting --niters 2000 --jump_type read --batch_size 30 --nsave 20 --evnt_align --restart > self_inhibiting/7612c40.log 2>self_inhibiting/7612c40.error &
taskset --cpu-list 2 python tnode.py --dataset powerlaw_hawkes --niters 2000 --jump_type read --batch_size 30 --nsave 20 --evnt_align --restart > powerlaw_hawkes/7612c40.log 2>powerlaw_hawkes/7612c40.error &
