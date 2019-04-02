run:
	taskset --cpu-list 0 python point_processes.py --dataset exponential_hawkes --niters 900 --jump_type read --batch_size 30 --nsave 10 --num_validation 20 --evnt_align > exponential_hawkes/${COMMIT}.log 2>exponential_hawkes/${COMMIT}.error &
	taskset --cpu-list 1 python point_processes.py --dataset self_inhibiting    --niters 900 --jump_type read --batch_size 30 --nsave 10 --num_validation 20 --evnt_align > self_inhibiting/${COMMIT}.log    2>self_inhibiting/${COMMIT}.error &
	taskset --cpu-list 2 python point_processes.py --dataset powerlaw_hawkes    --niters 900 --jump_type read --batch_size 30 --nsave 10 --num_validation 20 --evnt_align > powerlaw_hawkes/${COMMIT}.log    2>powerlaw_hawkes/${COMMIT}.error &

clean:
	rm exponential_hawkes/*
	rm self_inhibiting/*
	rm powerlaw_hawkes/*
