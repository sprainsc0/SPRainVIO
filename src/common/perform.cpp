#include <common/perform.h>
#include <common/singal_link.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
/**
 * Header common to all counters.
 */
struct perf_ctr_header {
	sq_entry_t		link;	/**< list linkage */
	enum perf_counter_type	type;	/**< counter type */
	const char		*name;	/**< counter name */
};

/**
 * PC_EVENT counter.
 */
struct perf_ctr_count {
	struct perf_ctr_header	hdr;
	uint64_t		event_count;
};

/**
 * PC_ELAPSED counter.
 */
struct perf_ctr_elapsed {
	struct perf_ctr_header	hdr;
	uint64_t		event_count;
	uint64_t		time_start;
	uint64_t		time_total;
	uint32_t		time_least;
	uint32_t		time_most;
	double			mean;
	double			M2;
};

/**
 * PC_INTERVAL counter.
 */
struct perf_ctr_interval {
	struct perf_ctr_header	hdr;
	uint64_t		event_count;
	uint64_t		time_event;
	uint64_t		time_first;
	uint64_t		time_last;
	uint32_t		time_least;
	uint32_t		time_most;
	double			mean;
	double			M2;
};

/**
 * List of all known counters.
 */
static sq_queue_t	perf_counters = { nullptr, nullptr };

perf_counter_t
perf_alloc(enum perf_counter_type type, const char *name)
{
	perf_counter_t ctr = nullptr;

	switch (type) {
	case PC_COUNT:
		ctr = (perf_counter_t)calloc(sizeof(struct perf_ctr_count), 1);
		break;

	case PC_ELAPSED:
		ctr = (perf_counter_t)calloc(sizeof(struct perf_ctr_elapsed), 1);
		break;

	case PC_INTERVAL:
		ctr = (perf_counter_t)calloc(sizeof(struct perf_ctr_interval), 1);

		break;

	default:
		break;
	}

	if (ctr != nullptr) {
		ctr->type = type;
		ctr->name = name;
		sq_addfirst(&ctr->link, &perf_counters);
	}

	return ctr;
}

perf_counter_t
perf_alloc_once(enum perf_counter_type type, const char *name)
{
	perf_counter_t handle = (perf_counter_t)sq_peek(&perf_counters);

	while (handle != nullptr) {
		if (!strcmp(handle->name, name)) {
			if (type == handle->type) {
				/* they are the same counter */
				return handle;

			} else {
				/* same name but different type, assuming this is an error and not intended */
				return nullptr;
			}
		}
		handle = (perf_counter_t)sq_next(&handle->link);
	}

	/* if the execution reaches here, no existing counter of that name was found */
	return perf_alloc(type, name);
}

void
perf_free(perf_counter_t handle)
{
	if (handle == nullptr) {
		return;
	}

	sq_rem(&handle->link, &perf_counters);
	free(handle);
}

void
perf_count(perf_counter_t handle, uint64_t time_us)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_COUNT:
		((struct perf_ctr_count *)handle)->event_count++;
		break;

	case PC_INTERVAL: {
			struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
			uint64_t now = time_us;

			switch (pci->event_count) {
			case 0:
				pci->time_first = now;
				break;

			case 1:
				pci->time_least = (uint32_t)(now - pci->time_last);
				pci->time_most = (uint32_t)(now - pci->time_last);
				pci->mean = pci->time_least / 1e6;
				pci->M2 = 0;
				break;

			default: {
					uint64_t interval = now - pci->time_last;

					if ((uint32_t)interval < pci->time_least) {
						pci->time_least = (uint32_t)interval;
					}

					if ((uint32_t)interval > pci->time_most) {
						pci->time_most = (uint32_t)interval;
					}

					// maintain mean and variance of interval in seconds
					// Knuth/Welford recursive mean and variance of update intervals (via Wikipedia)
					double dt = interval / 1e6;
					double delta_intvl = dt - pci->mean;
					pci->mean += delta_intvl / pci->event_count;
					pci->M2 += delta_intvl * (dt - pci->mean);
					break;
				}
			}

			pci->time_last = now;
			pci->event_count++;
			break;
		}

	default:
		break;
	}
}

uint32_t
perf_interval(perf_counter_t handle)
{
	struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
	uint32_t avg = (pci->event_count == 0) ? 0 : (unsigned long long)(pci->time_last - pci->time_first) / pci->event_count;
	return avg;
}

void
perf_begin(perf_counter_t handle, uint64_t time_us)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_ELAPSED:
		((struct perf_ctr_elapsed *)handle)->time_start = time_us;
		break;

	default:
		break;
	}
}

int32_t
perf_end(perf_counter_t handle, uint64_t time_us)
{
	if (handle == nullptr) {
		return 0;
	}
	int32_t elapsed = 0;
	switch (handle->type) {
	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;

			if (pce->time_start != 0) {
				elapsed = time_us - pce->time_start;

				if (elapsed >= 0) {

					pce->event_count++;
					pce->time_total += elapsed;

					if ((pce->time_least > (uint32_t)elapsed) || (pce->time_least == 0)) {
						pce->time_least = elapsed;
					}

					if (pce->time_most < (uint32_t)elapsed) {
						pce->time_most = elapsed;
					}

					// maintain mean and variance of the elapsed time in seconds
					// Knuth/Welford recursive mean and variance of update intervals (via Wikipedia)
					double dt = elapsed / 1e6;
					double delta_intvl = dt - pce->mean;
					pce->mean += delta_intvl / pce->event_count;
					pce->M2 += delta_intvl * (dt - pce->mean);

					pce->time_start = 0;
				}
			}
		}
		break;

	default:
		break;
	}
	return elapsed;
}

int32_t
perf_get_elapsed(perf_counter_t handle, uint64_t time_us)
{
	if (handle == NULL) {
		return 0;
	}
	int32_t elapsed = 0;
	struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;

	if (pce->time_start != 0) {
		elapsed = time_us - pce->time_start;
	}
	return elapsed;
}

void
perf_set_elapsed(perf_counter_t handle, int64_t elapsed)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;

			if (elapsed >= 0) {

				pce->event_count++;
				pce->time_total += elapsed;

				if ((pce->time_least > (uint32_t)elapsed) || (pce->time_least == 0)) {
					pce->time_least = elapsed;
				}

				if (pce->time_most < (uint32_t)elapsed) {
					pce->time_most = elapsed;
				}

				// maintain mean and variance of the elapsed time in seconds
				// Knuth/Welford recursive mean and variance of update intervals (via Wikipedia)
				double dt = elapsed / 1e6;
				double delta_intvl = dt - pce->mean;
				pce->mean += delta_intvl / pce->event_count;
				pce->M2 += delta_intvl * (dt - pce->mean);

				pce->time_start = 0;
			}
		}
		break;

	default:
		break;
	}
}

void
perf_set_count(perf_counter_t handle, uint64_t count)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_COUNT: {
			((struct perf_ctr_count *)handle)->event_count = count;
		}
		break;

	default:
		break;
	}

}

void
perf_cancel(perf_counter_t handle)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;

			pce->time_start = 0;
		}
		break;

	default:
		break;
	}
}



void
perf_reset(perf_counter_t handle)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_COUNT:
		((struct perf_ctr_count *)handle)->event_count = 0;
		break;

	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;
			pce->event_count = 0;
			pce->time_start = 0;
			pce->time_total = 0;
			pce->time_least = 0;
			pce->time_most = 0;
			break;
		}

	case PC_INTERVAL: {
			struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
			pci->event_count = 0;
			pci->time_event = 0;
			pci->time_first = 0;
			pci->time_last = 0;
			pci->time_least = 0;
			pci->time_most = 0;
			break;
		}
	}
}

void
perf_print_counter(perf_counter_t handle)
{
	if (handle == nullptr) {
		return;
	}

	switch (handle->type) {
	case PC_COUNT:
        printf("%10s- events %10llu|\n",
			handle->name,
			(unsigned long long)((struct perf_ctr_count *)handle)->event_count);
		break;

	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;
			double rms = sqrt(pce->M2 / (pce->event_count - 1));
            
			printf("%10s- events %10llu|avg %8lluus|min %8lluus|max %8lluus|rms %3.3fus\n",
				handle->name,
				(unsigned long long)pce->event_count,
				(pce->event_count == 0) ? 0 : (unsigned long long)pce->time_total / pce->event_count,
				(unsigned long long)pce->time_least,
				(unsigned long long)pce->time_most,
				(double)(1e6 * rms));
			break;
		}

	case PC_INTERVAL: {
			struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
			double rms = sqrt(pci->M2 / (pci->event_count - 1));
			unsigned long long avg = (pci->event_count == 0) ? 0 : (unsigned long long)(pci->time_last - pci->time_first) / pci->event_count;

			printf("%10s- events %10llu|freq %4dHz|avg %8lluus|min %8lluus|max %8lluus|rms %3.3fus\n",
				handle->name,
				(unsigned long long)pci->event_count,
				(pci->event_count == 0) ? 0 : (int)(1000000 / avg),
				avg,
				(unsigned long long)pci->time_least,
				(unsigned long long)pci->time_most,
				(double)(1e6 * rms));
			break;
		}

	default:
		break;
	}
}

int
perf_print_counter_buffer(char *buffer, int length, perf_counter_t handle)
{
	int num_written = 0;

	if (handle == nullptr) {
		return 0;
	}

	switch (handle->type) {
	case PC_COUNT:
		num_written = snprintf(buffer, length, "%s: %llu events",
				       handle->name,
				       (unsigned long long)((struct perf_ctr_count *)handle)->event_count);
		break;

	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;
			double rms = sqrt(pce->M2 / (pce->event_count - 1));
			num_written = snprintf(buffer, length, "%s: %llu events, %lluus elapsed, %lluus avg, min %lluus max %lluus %5.3fus rms",
					       handle->name,
					       (unsigned long long)pce->event_count,
					       (unsigned long long)pce->time_total,
					       (pce->event_count == 0) ? 0 : (unsigned long long)pce->time_total / pce->event_count,
					       (unsigned long long)pce->time_least,
					       (unsigned long long)pce->time_most,
					       (double)(1e6 * rms));
			break;
		}

	case PC_INTERVAL: {
			struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
			double rms = sqrt(pci->M2 / (pci->event_count - 1));

			num_written = snprintf(buffer, length, "%s: %llu events, %lluus avg, min %lluus max %lluus %5.3fus rms",
					       handle->name,
					       (unsigned long long)pci->event_count,
					       (pci->event_count == 0) ? 0 : (unsigned long long)(pci->time_last - pci->time_first) / pci->event_count,
					       (unsigned long long)pci->time_least,
					       (unsigned long long)pci->time_most,
					       (double)(1e6 * rms));
			break;
		}

	default:
		break;
	}

	buffer[length - 1] = 0; // ensure 0-termination
	return num_written;
}

uint64_t
perf_event_count(perf_counter_t handle)
{
	if (handle == nullptr) {
		return 0;
	}

	switch (handle->type) {
	case PC_COUNT:
		return ((struct perf_ctr_count *)handle)->event_count;

	case PC_ELAPSED: {
			struct perf_ctr_elapsed *pce = (struct perf_ctr_elapsed *)handle;
			return pce->event_count;
		}

	case PC_INTERVAL: {
			struct perf_ctr_interval *pci = (struct perf_ctr_interval *)handle;
			return pci->event_count;
		}

	default:
		break;
	}

	return 0;
}

void
perf_print_all(uint8_t type)
{
	perf_counter_t handle = (perf_counter_t)sq_peek(&perf_counters);
    
	while (handle != nullptr) {
		if(handle->type == type) {
			perf_print_counter(handle);
		}
		handle = (perf_counter_t)sq_next(&handle->link);
	}
}

void perf_print(char *name)
{
	perf_counter_t handle = (perf_counter_t)sq_peek(&perf_counters);
	while (handle != nullptr) {
		if(!strcmp(handle->name, name)) {
			perf_print_counter(handle);
		}
		handle = (perf_counter_t)sq_next(&handle->link);
	}
}

void
perf_reset_all(void)
{
	perf_counter_t handle = (perf_counter_t)sq_peek(&perf_counters);

	while (handle != nullptr) {
		perf_reset(handle);
		handle = (perf_counter_t)sq_next(&handle->link);
	}
}