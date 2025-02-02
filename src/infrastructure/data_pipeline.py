import json
from typing import Dict, Any
from kafka import KafkaProducer, KafkaConsumer

class DataPipeline:
    def __init__(self, kafka_broker='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.consumer = KafkaConsumer(
            'user_activity_topic',
            bootstrap_servers=[kafka_broker],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

    def publish_user_data(self, user_data: Dict[str, Any]):
        """
        Publish user activity data to Kafka topic
        """
        try:
            self.producer.send('user_activity_topic', user_data)
            self.producer.flush()
        except Exception as e:
            print(f"Error publishing user data: {e}")

    def consume_user_data(self):
        """
        Consume user activity data from Kafka topic
        """
        for message in self.consumer:
            yield message.value