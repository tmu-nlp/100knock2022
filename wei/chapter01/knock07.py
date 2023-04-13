print(('%d時の%sは%.1f') % (12,'気温',26.5))
print('{0}時の{1}は{2}'.format('18','気温','15.3'))

'''example as'''
print('{}は今年{{25}}歳です。'.format('花子さん'))
info = {'name':'zhangsan', 'age':11}
print('His name is {name},and he\'s {age} years old.'.format(**info))