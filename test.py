import time

total_time = 1000 * 10
time_elapsed = 0

for i in range(0, 1000):
    print(time_elapsed, 'out of', total_time)
    print("sleeping")
    time_elapsed += 10
    time.sleep(10)
print('done')