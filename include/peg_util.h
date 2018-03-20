#ifndef
#define PEG_UTIL_H

template <typename V_T>
inline bool peg_CAS(V_T *ptr, V_T old_val, V_T new_val)
{
	if (1 == sizeof(V_T)) {

	} else if (4 == sizeof(V_T)) {

	} else if (8 == sizeof(V_T) && 8 == sizeof(long)) {

	}
}

#endif
