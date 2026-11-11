import { createWorkerHandler } from './index.js';

export default {
  fetch(request, env) {
    return createWorkerHandler({ bearerToken: env.WSP_BEARER_TOKEN }).fetch(request);
  }
};
