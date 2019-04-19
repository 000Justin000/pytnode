run_point_process:
	taskset --cpu-list 0 python point_processes.py --dataset poisson            --niters 3000 --jump_type read --batch_size 30 --nsave 50 --seed0 --evnt_align &
	taskset --cpu-list 1 python point_processes.py --dataset exponential_hawkes --niters 3000 --jump_type read --batch_size 30 --nsave 50 --seed0 --evnt_align &
	taskset --cpu-list 2 python point_processes.py --dataset self_inhibiting    --niters 3000 --jump_type read --batch_size 30 --nsave 50 --seed0 --evnt_align &
	taskset --cpu-list 3 python point_processes.py --dataset powerlaw_hawkes    --niters 3000 --jump_type read --batch_size 30 --nsave 50 --seed0 --evnt_align &

run_order_book:
	taskset --cpu-list 0 python order_book.py --niters 1500 --jump_type read --batch_size 20 --nsave 50 --paramr workspace/dataset\:book_order-pid\:9766/params.pth --restart --seed0 &

run_coupled_oscillators:
	taskset --cpu-list 0 python coupled_oscillators.py --niters 070000 --batch_size 300 --nsave 100 --seed0 &
