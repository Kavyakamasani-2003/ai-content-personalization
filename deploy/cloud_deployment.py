from kubernetes import client, config

class CloudDeployer:
    def __init__(self, cloud_provider='gcp'):
        """
        Initialize cloud deployment configuration
        
        Args:
            cloud_provider (str): Cloud provider (gcp, aws, azure)
        """
        self.cloud_provider = cloud_provider
        
        # Load Kubernetes configuration
        try:
            config.load_kube_config()
            self.k8s_client = client.CoreV1Api()
        except Exception as e:
            print(f"Kubernetes configuration error: {e}")
    
    def deploy_microservice(self, service_name, image, replicas=3):
        """
        Deploy a microservice to Kubernetes cluster
        
        Args:
            service_name (str): Name of the microservice
            image (str): Docker image for the microservice
            replicas (int): Number of replicas
        """
        # Deployment specification
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=service_name),
            spec={
                'replicas': replicas,
                'template': {
                    'metadata': {'labels': {'app': service_name}},
                    'spec': {
                        'containers': [{
                            'name': service_name,
                            'image': image
                        }]
                    }
                }
            }
        )
        
        # Create deployment
        self.k8s_client.create_namespaced_deployment(
            body=deployment, 
            namespace='default'
        )