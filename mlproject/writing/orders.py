from mlproject import settings
from django import template
from django.contrib.auth.models import Group
register = template.Library()
import simpy 

class Customer(object):
    def __init__(self, env, number):
        self.env = env
        self.number = number 
        #분포에 따라 customer 도착
        self.action = env.process(self.customer_generate())


    def customer_generate(self):
        for i in range(self.number):
            name = 'Customer-%s'%i            
            arrive = self.env.now
            print(name, '%8.3f' %arrive,'카페도착')

            #도착한 고객은 주문하러 이동 
            self.env.process(self.order_coffee(name, staff))
            
            interval_time = 10
            yield self.env.timeout(interval_time)

    
    def order_coffee(self, name, staff):
        #카운터 직원 요청 
        with staff.request() as req:
            yield req

            #직원에게 30초동안 주문 
            ordering_duration = 30
            yield self.env.timeout(ordering_duration)
            print(name, '%8.3f'%self.env.now, '주문완료')
            
        #주문한 고객은 커피 수령을 위해 대기
        yield self.env.process(self.wait_for_coffee(name))


    def wait_for_coffee(self, name):
        #30초 대기 후 커피 수령 
        waiting_duration = 30
        yield(self.env.timeout(waiting_duration))
        print(name, '%8.3f' %self.env.now,'커피수령')


print('coffee order')

env = simpy.Environment()
staff = simpy.Resource(env, capacity=2)
customer = Customer(env, 10)

env.run()