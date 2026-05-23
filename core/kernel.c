/*
 * Arthedain SNN Core Kernel
 * =========================
 * C implementation of forward pass + Hebbian update for real-time execution.
 * 
 * No Python runtime dependency. Suitable for:
 * - Real-time Linux with SCHED_FIFO
 * - ROS 2 nodes
 * - Bare-metal ARM (STM32H7, i.MX RT1176)
 * 
 * Compile: gcc -O3 -shared -fPIC -o kernel.so kernel.c
 */

#include <stdint.h>
#include <math.h>
#include <string.h>

/* Configuration constants (match Python defaults) */
#define V_THRESHOLD 1.0f
#define V_RESET 0.0f
#define TAU 20.0f
#define DT 1.0f
#define BETA_SURROGATE 10.0f
#define EPSILON 1e-6f

/* Fixed-point scaling (for integer mode) */
#define FP_FRAC_BITS 8
#define FP_SCALE (1 << FP_FRAC_BITS)
#define FP_SCALE_F ((float)FP_SCALE)

/*
 * LIF Neuron Step
 * ---------------
 * Computes: v = beta * v + I
 *           spike = (v >= v_th) ? 1 : 0
 *           v = spike ? v_reset : v
 * 
 * Returns spike count (0 or 1) per neuron.
 */
void lif_step(
    const float* current,      /* Input: input current [n_neurons] */
    float* v,                  /* In/Out: membrane potential [n_neurons] */
    float* spikes,             /* Out: spike vector [n_neurons] */
    int n_neurons,             /* Number of neurons */
    int refractory,            /* Refractory period (not used in simple version) */
    float tau,                 /* Membrane time constant */
    float v_th,              /* Spike threshold */
    float v_reset            /* Reset potential */
) {
    float beta = expf(-DT / tau);
    
    for (int i = 0; i < n_neurons; i++) {
        /* Update membrane potential */
        v[i] = beta * v[i] + current[i];
        
        /* Generate spike */
        spikes[i] = (v[i] >= v_th) ? 1.0f : 0.0f;
        
        /* Reset if spiked */
        if (spikes[i] > 0.5f) {
            v[i] = v_reset;
        }
    }
}

/*
 * Fast Sigmoid Surrogate Gradient (d_LIF)
 * ----------------------------------------
 * Computes: d_LIF = 1 / (1 + beta * |v - theta|)
 * 
 * Used for three-factor Hebbian rule and confidence estimation.
 */
void compute_d_lif(
    const float* v,            /* Membrane potentials [n_neurons] */
    float* d_lif,              /* Out: d_LIF values [n_neurons] */
    int n_neurons,
    float beta,
    float v_th
) {
    for (int i = 0; i < n_neurons; i++) {
        float dist = fabsf(v[i] - v_th);
        d_lif[i] = 1.0f / (1.0f + beta * dist);
    }
}

/*
 * Forward Pass
 * ------------
 * Computes: current = W_in @ x + W_rec @ s_prev
 *           spikes = LIF_step(current)
 *           output = W_out @ spikes
 */
void snn_forward(
    /* Inputs */
    const float* x,            /* Input spike counts [input_size] */
    const float* s_prev,       /* Previous spikes [hidden_size] */
    
    /* Parameters */
    const float* W_in,         /* [hidden_size][input_size] */
    const float* W_rec,        /* [hidden_size][hidden_size] */
    const float* W_out,        /* [output_size][hidden_size] */
    
    /* State */
    float* v,                  /* Membrane potentials [hidden_size] */
    
    /* Outputs */
    float* spikes,             /* Current spikes [hidden_size] */
    float* output,             /* Output prediction [output_size] */
    
    /* Dimensions */
    int input_size,
    int hidden_size,
    int output_size
) {
    /* Compute input current: W_in @ x + W_rec @ s_prev */
    float* current = (float*)malloc(hidden_size * sizeof(float));
    
    for (int i = 0; i < hidden_size; i++) {
        current[i] = 0.0f;
        
        /* W_in @ x */
        for (int j = 0; j < input_size; j++) {
            current[i] += W_in[i * input_size + j] * x[j];
        }
        
        /* + W_rec @ s_prev */
        for (int j = 0; j < hidden_size; j++) {
            current[i] += W_rec[i * hidden_size + j] * s_prev[j];
        }
    }
    
    /* LIF step */
    lif_step(current, v, spikes, hidden_size, 2, TAU, V_THRESHOLD, V_RESET);
    
    /* Readout: W_out @ spikes */
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W_out[i * hidden_size + j] * spikes[j];
        }
    }
    
    free(current);
}

/*
 * Dual-Timescale Hebbian Update
 * ------------------------------
 * Computes eligibility traces and weight updates.
 * 
 * e_fast = lambda_f * e_fast + outer(post, pre)
 * e_slow = lambda_s * e_slow + outer(post, pre)
 * E = alpha * e_fast + (1-alpha) * e_slow
 */
void hebbian_update(
    /* Inputs */
    const float* pre,          /* Presynaptic spikes [n_pre] */
    const float* post,         /* Postsynaptic spikes [n_post] */
    
    /* State */
    float* e_fast,             /* Fast trace [n_post][n_pre] (flat) */
    float* e_slow,             /* Slow trace [n_post][n_pre] (flat) */
    
    /* Parameters */
    float lambda_fast,         /* Fast decay: exp(-dt/tau_fast) */
    float lambda_slow,         /* Slow decay: exp(-dt/tau_slow) */
    float alpha,               /* Fast trace mixing weight */
    
    /* Output */
    float* E,                  /* Combined eligibility [n_post][n_pre] (flat) */
    
    /* Dimensions */
    int n_pre,
    int n_post
) {
    float beta = 1.0f - alpha;
    
    for (int i = 0; i < n_post; i++) {
        for (int j = 0; j < n_pre; j++) {
            int idx = i * n_pre + j;
            
            /* Outer product */
            float outer = post[i] * pre[j];
            
            /* Update traces */
            e_fast[idx] = lambda_fast * e_fast[idx] + outer;
            e_slow[idx] = lambda_slow * e_slow[idx] + outer;
            
            /* Combine */
            E[idx] = alpha * e_fast[idx] + beta * e_slow[idx];
        }
    }
}

/*
 * RMS Normalization with 256-entry LUT
 * -------------------------------------
 * Uses lookup table for 1/sqrt(mean(v^2) + epsilon)
 */
static float rms_lut[256];
static int lut_initialized = 0;

void init_rms_lut(void) {
    if (lut_initialized) return;
    
    float max_rms = 128.0f;  /* For INT16 range */
    for (int i = 0; i < 256; i++) {
        float rms = max_rms * i / 255.0f;
        rms_lut[i] = 1.0f / sqrtf(rms + EPSILON);
    }
    
    lut_initialized = 1;
}

float rms_normalize_lut(const float* v, float* out, int n) {
    /* Compute mean squared */
    float mean_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        mean_sq += v[i] * v[i];
    }
    mean_sq /= n;
    
    float rms = sqrtf(mean_sq);
    
    /* LUT lookup */
    int idx = (int)((rms / 128.0f) * 255.0f);
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    
    float inv_rms = rms_lut[idx];
    
    /* Normalize */
    for (int i = 0; i < n; i++) {
        out[i] = v[i] * inv_rms;
    }
    
    return rms;
}

/*
 * Complete Training Step
 * ----------------------
 * One timestep of online learning with all components.
 */
void training_step(
    /* Inputs */
    const float* x,
    const float* target,
    
    /* Model parameters */
    float* W_in,
    float* W_rec,
    float* W_out,
    float* b_out,
    
    /* State */
    float* v,                  /* Membrane potentials */
    float* s_prev,             /* Previous spikes */
    float* e_fast,
    float* e_slow,
    
    /* Learning rates */
    float lr_readout,
    float lr_recurrent,
    
    /* Hyperparameters */
    float tau_fast,
    float tau_slow,
    float alpha,
    
    /* Outputs */
    float* output,
    float* error,
    
    /* Dimensions */
    int input_size,
    int hidden_size,
    int output_size
) {
    /* Decay factors */
    float lambda_fast = expf(-DT / tau_fast);
    float lambda_slow = expf(-DT / tau_slow);
    
    /* Forward pass */
    float* spikes = (float*)malloc(hidden_size * sizeof(float));
    float* current = (float*)malloc(hidden_size * sizeof(float));
    
    /* Compute current */
    for (int i = 0; i < hidden_size; i++) {
        current[i] = 0.0f;
        for (int j = 0; j < input_size; j++) {
            current[i] += W_in[i * input_size + j] * x[j];
        }
        for (int j = 0; j < hidden_size; j++) {
            current[i] += W_rec[i * hidden_size + j] * s_prev[j];
        }
    }
    
    /* LIF step */
    lif_step(current, v, spikes, hidden_size, 2, TAU, V_THRESHOLD, V_RESET);
    
    /* Readout */
    for (int i = 0; i < output_size; i++) {
        output[i] = b_out[i];
        for (int j = 0; j < hidden_size; j++) {
            output[i] += W_out[i * hidden_size + j] * spikes[j];
        }
    }
    
    /* Error */
    for (int i = 0; i < output_size; i++) {
        error[i] = target[i] - output[i];
    }
    
    /* Hebbian update */
    float* E = (float*)malloc(hidden_size * hidden_size * sizeof(float));
    hebbian_update(s_prev, spikes, e_fast, e_slow, 
                   lambda_fast, lambda_slow, alpha, E,
                   hidden_size, hidden_size);
    
    /* Weight updates with clipping */
    float max_update = 0.1f;
    
    /* Readout update */
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float update = lr_readout * error[i] * spikes[j];
            if (update > max_update) update = max_update;
            if (update < -max_update) update = -max_update;
            W_out[i * hidden_size + j] += update;
        }
        b_out[i] += fmaxf(fminf(lr_readout * error[i], max_update), -max_update);
    }
    
    /* Recurrent update */
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            float update = lr_recurrent * E[i * hidden_size + j];
            if (update > max_update) update = max_update;
            if (update < -max_update) update = -max_update;
            W_rec[i * hidden_size + j] += update;
        }
    }
    
    /* Update previous spikes */
    memcpy(s_prev, spikes, hidden_size * sizeof(float));
    
    /* Cleanup */
    free(spikes);
    free(current);
    free(E);
}

/*
 * Fixed-Point Versions (INT16)
 * =============================
 * Integer-only implementation for FPGA/microcontroller.
 */

void lif_step_int16(
    const int16_t* current,
    int16_t* v,
    int16_t* spikes,
    int n_neurons,
    int beta_int,              /* Fixed-point beta (scaled by FP_SCALE) */
    int v_th_int,
    int v_reset_int
) {
    for (int i = 0; i < n_neurons; i++) {
        /* v = (beta * v) >> frac_bits + current */
        int32_t v_new = ((int32_t)beta_int * (int32_t)v[i]) >> FP_FRAC_BITS;
        v_new += current[i];
        
        /* Clip to int16 range */
        if (v_new > 32767) v_new = 32767;
        if (v_new < -32768) v_new = -32768;
        
        /* Spike detection */
        spikes[i] = (v_new >= v_th_int) ? 1 : 0;
        
        /* Reset */
        v[i] = (spikes[i] > 0) ? v_reset_int : (int16_t)v_new;
    }
}

/*
 * Python Interface (via ctypes)
 * =============================
 * Export functions for Python binding.
 */

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void kernel_lif_step(
    const float* current,
    float* v,
    float* spikes,
    int n_neurons
) {
    lif_step(current, v, spikes, n_neurons, 2, TAU, V_THRESHOLD, V_RESET);
}

EXPORT void kernel_forward(
    const float* x,
    const float* s_prev,
    const float* W_in,
    const float* W_rec,
    const float* W_out,
    float* v,
    float* spikes,
    float* output,
    int input_size,
    int hidden_size,
    int output_size
) {
    snn_forward(x, s_prev, W_in, W_rec, W_out, v, 
                spikes, output, input_size, hidden_size, output_size);
}

EXPORT void kernel_training_step(
    const float* x,
    const float* target,
    float* W_in,
    float* W_rec,
    float* W_out,
    float* b_out,
    float* v,
    float* s_prev,
    float* e_fast,
    float* e_slow,
    float lr_readout,
    float lr_recurrent,
    float tau_fast,
    float tau_slow,
    float alpha,
    float* output,
    float* error,
    int input_size,
    int hidden_size,
    int output_size
) {
    training_step(x, target, W_in, W_rec, W_out, b_out,
                  v, s_prev, e_fast, e_slow,
                  lr_readout, lr_recurrent,
                  tau_fast, tau_slow, alpha,
                  output, error,
                  input_size, hidden_size, output_size);
}

EXPORT void kernel_init_rms_lut(void) {
    init_rms_lut();
}
