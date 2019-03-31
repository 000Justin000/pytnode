taskset --cpu-list 0 python tnode.py --dataset exponential_hawkes --niters 2000 --jump_type read --batch_size 30 --nsave 1 --evnt_align > exponential_hawkes/$1.log 2>exponential_hawkes/$1.error &
taskset --cpu-list 1 python tnode.py --dataset self_inhibiting --niters 2000 --jump_type read --batch_size 30 --nsave 1 --evnt_align > self_inhibiting/$1.log 2>self_inhibiting/$1.error &
taskset --cpu-list 2 python tnode.py --dataset powerlaw_hawkes --niters 2000 --jump_type read --batch_size 30 --nsave 1 --evnt_align > powerlaw_hawkes/$1.log 2>powerlaw_hawkes/$1.error &
